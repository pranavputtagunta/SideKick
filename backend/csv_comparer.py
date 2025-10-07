import pandas as pd
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


def normalization_fallback_bbox(df):
    """Fallback normalization based on bounding box when torso landmarks are missing.

    Returns a per-frame normalization factor (scalar) to divide coordinates by.
    """
    # Find all coordinate columns
    coord_cols = [c for c in df.columns if c.startswith(('x_', 'y_'))]
    if not coord_cols:
        return np.ones(len(df))

    # Compute bbox size per frame (max range in x or y)
    xs = df[[c for c in df.columns if c.startswith('x_')]].values
    ys = df[[c for c in df.columns if c.startswith('y_')]].values
    x_range = np.nanmax(xs, axis=1) - np.nanmin(xs, axis=1)
    y_range = np.nanmax(ys, axis=1) - np.nanmin(ys, axis=1)
    bbox_size = np.maximum(x_range, y_range)
    bbox_size[bbox_size == 0] = 1.0
    return bbox_size


def normalize_skeleton(df):
    """Normalizes the skeleton based on torso size, or falls back to bbox size.

    This function divides all x/y/z coordinates by a per-frame scalar to normalize scale.
    """
    # These landmarks are part of the BlazePose model
    left_shoulder = ['x_pose_LEFT_SHOULDER', 'y_pose_LEFT_SHOULDER', 'z_pose_LEFT_SHOULDER']
    right_shoulder = ['x_pose_RIGHT_SHOULDER', 'y_pose_RIGHT_SHOULDER', 'z_pose_RIGHT_SHOULDER']
    left_hip = ['x_pose_LEFT_HIP', 'y_pose_LEFT_HIP', 'z_pose_LEFT_HIP']
    right_hip = ['x_pose_RIGHT_HIP', 'y_pose_RIGHT_HIP', 'z_pose_RIGHT_HIP']

    required_cols = left_shoulder + right_shoulder + left_hip + right_hip
    use_torso = all(col in df.columns for col in required_cols)

    if use_torso:
        # Calculate torso size for each frame
        shoulder_dist = np.linalg.norm(df[left_shoulder].values - df[right_shoulder].values, axis=1)
        hip_dist = np.linalg.norm(df[left_hip].values - df[right_hip].values, axis=1)
        torso_height = np.linalg.norm(df[left_shoulder].values - df[left_hip].values, axis=1)
        # Average torso size, avoiding division by zero
        normalization_factor = (shoulder_dist + hip_dist + torso_height) / 3.0
        normalization_factor[normalization_factor == 0] = 1.0
    else:
        normalization_factor = normalization_fallback_bbox(df)

    # Get all landmark columns (x/y/z)
    landmark_cols = [col for col in df.columns if col.startswith(('x_', 'y_', 'z_'))]

    # Normalize all landmark coordinates per-frame
    for col in landmark_cols:
        # broadcast dividing column vector by normalization_factor
        df[col] = df[col].values / normalization_factor

    return df


def calculate_distance(expert_df, user_df):
    """Calculates the Euclidean distance between aligned landmarks."""
    # Find common landmarks (based on x_ columns)
    expert_landmarks = {col.replace('x_', '').replace('y_', '').replace('z_', '') for col in expert_df.columns if col.startswith('x_')}
    user_landmarks = {col.replace('x_', '').replace('y_', '').replace('z_', '') for col in user_df.columns if col.startswith('x_')}
    common_landmarks = sorted(list(expert_landmarks.intersection(user_landmarks)))

    if not common_landmarks:
        raise ValueError("No common landmarks found between the two CSV files.")

    # Prepare data for DTW: flatten per-frame (x1,y1,z1,x2,y2,z2,...)
    expert_series = expert_df[[f'{axis}_{lm}' for lm in common_landmarks for axis in ['x', 'y', 'z']]].values
    user_series = user_df[[f'{axis}_{lm}' for lm in common_landmarks for axis in ['x', 'y', 'z']]].values

    # DTW alignment (euclidean distance as default)
    alignment = dtw(expert_series, user_series, keep_internals=True)

    # Get aligned indices
    expert_indices = alignment.index1
    user_indices = alignment.index2

    # Calculate Euclidean distance for each aligned frame pair
    distances = []
    for exp_idx, usr_idx in zip(expert_indices, user_indices):
        frame_dist = 0.0
        valid_count = 0
        for lm in common_landmarks:
            try:
                exp_point = expert_df.loc[exp_idx, [f'x_{lm}', f'y_{lm}', f'z_{lm}']].values.astype(float)
                usr_point = user_df.loc[usr_idx, [f'x_{lm}', f'y_{lm}', f'z_{lm}']].values.astype(float)
            except Exception:
                continue
            # If either point contains NaNs, skip
            if np.isnan(exp_point).any() or np.isnan(usr_point).any():
                continue
            frame_dist += np.linalg.norm(exp_point - usr_point)
            valid_count += 1
        if valid_count > 0:
            distances.append(frame_dist / valid_count)

    if len(distances) == 0:
        raise ValueError('No valid aligned distances were computed.')

    return float(np.mean(distances))


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

        # Generate accuracy score (simple linear conversion)
        accuracy = 1.0 - avg_distance
        accuracy = max(0.0, min(1.0, accuracy))  # Clamp between 0 and 1
        accuracy_percentage = accuracy * 100.0

        print(f"Average Normalized Distance: {avg_distance:.5f}")
        print(f"Accuracy Score: {accuracy_percentage:.2f}%")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
