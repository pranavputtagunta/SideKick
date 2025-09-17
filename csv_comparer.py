import pandas as pd
import numpy as np
from dtw import dtw
import argparse

def load_and_preprocess_data(csv_path):
    """Loads landmark data from a CSV and preprocesses it."""
    df = pd.read_csv(csv_path)
    
    # Pivot the table to have landmarks as columns
    df_pivot = df.pivot_table(index='frame_number', columns='landmark_name', values=['x', 'y', 'z'])
    
    # Flatten the multi-level columns
    df_pivot.columns = [f'{val}_{col}' for val, col in df_pivot.columns]
    df_pivot = df_pivot.reset_index()
    
    # Fill missing values
    df_pivot = df_pivot.ffill().bfill()
    
    return df_pivot

def normalize_skeleton(df):
    """Normalizes the skeleton based on torso size."""
    # These landmarks are part of the BlazePose model
    left_shoulder = ['x_pose_LEFT_SHOULDER', 'y_pose_LEFT_SHOULDER', 'z_pose_LEFT_SHOULDER']
    right_shoulder = ['x_pose_RIGHT_SHOULDER', 'y_pose_RIGHT_SHOULDER', 'z_pose_RIGHT_SHOULDER']
    left_hip = ['x_pose_LEFT_HIP', 'y_pose_LEFT_HIP', 'z_pose_LEFT_HIP']
    right_hip = ['x_pose_RIGHT_HIP', 'y_pose_RIGHT_HIP', 'z_pose_RIGHT_HIP']

    # Check if all required columns are present
    required_cols = left_shoulder + right_shoulder + left_hip + right_hip
    if not all(col in df.columns for col in required_cols):
        raise ValueError("CSV file does not contain the required pose landmarks for normalization (shoulders and hips).")

    # Calculate torso size for each frame
    shoulder_dist = np.linalg.norm(df[left_shoulder].values - df[right_shoulder].values, axis=1)
    hip_dist = np.linalg.norm(df[left_hip].values - df[right_hip].values, axis=1)
    torso_height = np.linalg.norm(df[left_shoulder].values - df[left_hip].values, axis=1)
    
    # Average torso size, avoiding division by zero
    normalization_factor = (shoulder_dist + hip_dist + torso_height) / 3
    normalization_factor[normalization_factor == 0] = 1

    # Get all landmark columns
    landmark_cols = [col for col in df.columns if col.startswith(('x_', 'y_', 'z_'))]
    
    # Normalize all landmark coordinates
    for col in landmark_cols:
        # The normalization is applied based on the dimension (x, y, or z)
        # This is a simplified approach; more complex methods might be needed for rotation
        df[col] = df[col] / normalization_factor

    return df

def calculate_distance(expert_df, user_df):
    """Calculates the Euclidean distance between aligned landmarks."""
    # Find common landmarks
    expert_landmarks = {col.replace('x_', '').replace('y_', '').replace('z_', '') for col in expert_df.columns if col.startswith('x_')}
    user_landmarks = {col.replace('x_', '').replace('y_', '').replace('z_', '') for col in user_df.columns if col.startswith('x_')}
    common_landmarks = list(expert_landmarks.intersection(user_landmarks))

    if not common_landmarks:
        raise ValueError("No common landmarks found between the two CSV files.")

    # Prepare data for DTW
    expert_series = expert_df[[f'{axis}_{lm}' for lm in common_landmarks for axis in ['x', 'y', 'z']]].values
    user_series = user_df[[f'{axis}_{lm}' for lm in common_landmarks for axis in ['x', 'y', 'z']]].values

    # DTW alignment
    alignment = dtw(expert_series, user_series, keep_internals=True)

    # Get aligned indices
    expert_indices = alignment.index1
    user_indices = alignment.index2

    # Calculate Euclidean distance for each aligned frame pair
    distances = []
    for i in range(len(expert_indices)):
        exp_idx = expert_indices[i]
        usr_idx = user_indices[i]
        
        frame_dist = 0
        for lm in common_landmarks:
            exp_point = expert_df.loc[exp_idx, [f'x_{lm}', f'y_{lm}', f'z_{lm}']].values
            usr_point = user_df.loc[usr_idx, [f'x_{lm}', f'y_{lm}', f'z_{lm}']].values
            frame_dist += np.linalg.norm(exp_point - usr_point)
        
        distances.append(frame_dist / len(common_landmarks))

    return np.mean(distances)

def main():
    parser = argparse.ArgumentParser(description='Compare two landmark CSV files using DTW.')
    parser.add_argument('expert_csv', type=str, help='Path to the expert\'s landmark CSV file.')
    parser.add_argument('user_csv', type=str, help='Path to the user\'s landmark CSV file.')
    args = parser.parse_args()

    try:
        # Load and preprocess data
        expert_data = load_and_preprocess_data(args.expert_csv)
        user_data = load_and_preprocess_data(args.user_csv)

        # Normalize skeletons
        expert_normalized = normalize_skeleton(expert_data)
        user_normalized = normalize_skeleton(user_data)

        # Calculate average distance
        avg_distance = calculate_distance(expert_normalized, user_normalized)

        # Generate accuracy score
        accuracy = 1 - avg_distance
        accuracy = max(0, min(1, accuracy)) # Clamp between 0 and 1
        accuracy_percentage = accuracy * 100

        print(f"Average Normalized Distance: {avg_distance:.4f}")
        print(f"Accuracy Score: {accuracy_percentage:.2f}%")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
