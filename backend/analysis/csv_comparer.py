import pandas as pd
import pandas as pd
import numpy as np
from dtw import dtw
import json

def load_and_preprocess_data(csv_path):
    """Loads landmark data from a CSV and preprocesses it."""
    df = pd.read_csv(csv_path)

    timestamps = df[['frame_number', 'timestamp_ms']].drop_duplicates().set_index('frame_number')

    # Pivot the table to have landmarks as columns
    df_pivot = df.pivot_table(index='frame_number', columns='landmark_name', values=['x', 'y', 'z'])

    # Flatten the multi-level columns
    df_pivot.columns = [f'{val}_{col}' for val, col in df_pivot.columns]
    df_pivot = df_pivot.reset_index()

    df_pivot = pd.merge(df_pivot, timestamps, on= 'frame_number')

    # Fill missing values
    df_pivot = df_pivot.ffill().bfill()

    return df_pivot

def get_discrepancy_direction(expert_point, user_point):
    """Determines the direction of discrepancy between expert and user points.

    Retruns a list of directions for each axis (x, y, z).
    """
    diff = expert_point - user_point
    directions = []

    threshold = 0.05  # Define a threshold to consider significant difference

    # May need to change based on coordinate system (testing required) 
    if diff[0] > threshold:
        directions.append('right')
    elif diff[0] < -threshold:
        directions.append('left')
    if diff[1] > threshold:
        directions.append('down')
    elif diff[1] < -threshold:
        directions.append('up')
    if diff[2] > threshold:
        directions.append('backward')
    elif diff[2] < -threshold:
        directions.append('forward')

    return directions

def calculate_detailed_distance(expert_df, user_df, deviant_landmarks=50):
    """"
    Calculates detailed per-landmark Euclidean distances. 
    Also finds the peak error frames for feedback. 
    """
    # Find common landmark names between expert and user (based on x_ columns)
    expert_landmarks = {col.split('_', 1)[1] for col in expert_df.columns if col.startswith('x_')}
    user_landmarks = {col.split('_', 1)[1] for col in user_df.columns if col.startswith('x_')}
    common_landmarks = sorted(list(expert_landmarks.intersection(user_landmarks)))

    if not common_landmarks:
        raise ValueError("No common landmarks found between the two CSV files.")
    
    # Prepare data for DTW: flatten per-frame (x1,y1,z1,x2,y2,z2,...)
    expert_series = expert_df[[f'{axis}_{lm}' for lm in common_landmarks for axis in ['x', 'y', 'z']]].values
    user_series = user_df[[f'{axis}_{lm}' for lm in common_landmarks for axis in ['x', 'y', 'z']]].values

    # DTW alignment
    alignment = dtw(expert_series, user_series, keep_internals=True)
    expert_indices = alignment.index1
    user_indices = alignment.index2

    # Data structures to hold distances
    landmark_distances = {lm: [] for lm in common_landmarks}
    frame_distances = []

    # Loop through aligned frames
    for exp_idx, usr_idx in zip(expert_indices, user_indices):
        total_frame_dist = 0.0
        valid_landmarks_in_frame = 0
        for lm in common_landmarks: 
            try: 
                exp_point = expert_df.loc[exp_idx, [f'x_{lm}', f'y_{lm}', f'z_{lm}']].values.astype(float)
                usr_point = user_df.loc[usr_idx, [f'x_{lm}', f'y_{lm}', f'z_{lm}']].values.astype(float)
            except KeyError:
                continue

            if np.isnan(exp_point).any() or np.isnan(usr_point).any():
                continue

            dist = np.linalg.norm(exp_point - usr_point)
            landmark_distances[lm].append(dist)
            total_frame_dist += dist
            valid_landmarks_in_frame += 1

        if valid_landmarks_in_frame > 0:
            avg_frame_dist = total_frame_dist / valid_landmarks_in_frame
            frame_distances.append((avg_frame_dist, user_df.loc[usr_idx, 'timestamp_ms'], usr_idx, exp_idx))

    if not frame_distances:
        raise ValueError('No valid aligned distances were computed.')
    
    # Highest error frame
    peak_error_dist, peak_timestamp, peak_usr_idx, peak_exp_idx = max(frame_distances, key=lambda x: x[0])

    # Calculate final per-landmark average distances
    per_landmark_avg_dist = {
        lm: np.mean(dists) for lm, dists in landmark_distances.items() if dists
    }

    # Top most deviant landmarks
    top_deviant_landmarks = sorted(per_landmark_avg_dist.items(), key=lambda x: x[1], reverse=True)[:deviant_landmarks]

    # Get details of top deviant landmarks at peak error frame
    discrepancies = []
    for lm, magnitude in top_deviant_landmarks:
        try:
            exp_point = expert_df.loc[peak_exp_idx, [f'x_{lm}', f'y_{lm}', f'z_{lm}']].values.astype(float)
            usr_point = user_df.loc[peak_usr_idx, [f'x_{lm}', f'y_{lm}', f'z_{lm}']].values.astype(float)
            if not np.isnan(exp_point).any() and not np.isnan(usr_point).any():
                directions = get_discrepancy_direction(exp_point, usr_point)
                discrepancies.append({
                    'landmark': lm.replace('pose_', ''),
                    'magnitude': round(magnitude, 5),
                    'directions': directions
                })

        except (KeyError, IndexError):
            continue

    overall_avg_distance = float(np.mean([dist for dist, _, _, _ in frame_distances]))

    return {
        'overall_average_distance': overall_avg_distance,
        'discrepancies': discrepancies,
        'peak_error': {
            'distance': round(peak_error_dist, 5),
            'timestamp_ms': int(peak_timestamp)
        }
    }

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
    """
    Normalizes the skeleton by centering it on the hips and scaling based on 
    the 2D torso size. This makes the analysis invariant to position and scale.
    """
    # Define torso landmarks for calculating the center and scale
    left_shoulder_2d = ['x_pose_LEFT_SHOULDER', 'y_pose_LEFT_SHOULDER']
    right_shoulder_2d = ['x_pose_RIGHT_SHOULDER', 'y_pose_RIGHT_SHOULDER']
    left_hip_2d = ['x_pose_LEFT_HIP', 'y_pose_LEFT_HIP']
    right_hip_2d = ['x_pose_RIGHT_HIP', 'y_pose_RIGHT_HIP']

    # Check if all required 2D torso landmarks are present
    required_cols = left_shoulder_2d + right_shoulder_2d + left_hip_2d + right_hip_2d
    use_torso = all(col in df.columns for col in required_cols)

    if not use_torso:
        print("Warning: Torso landmarks not found. Skipping normalization.")
        return df

    # --- 1. Centering (using 3D center) ---
    hip_center_x = (df['x_pose_LEFT_HIP'] + df['x_pose_RIGHT_HIP']) / 2
    hip_center_y = (df['y_pose_LEFT_HIP'] + df['y_pose_RIGHT_HIP']) / 2
    hip_center_z = (df['z_pose_LEFT_HIP'] + df['z_pose_RIGHT_HIP']) / 2

    x_cols = [col for col in df.columns if col.startswith('x_')]
    y_cols = [col for col in df.columns if col.startswith('y_')]
    z_cols = [col for col in df.columns if col.startswith('z_')]

    # Subtract the hip center from all landmarks for each frame
    for col in x_cols:
        df[col] = df[col] - hip_center_x
    for col in y_cols:
        df[col] = df[col] - hip_center_y
    for col in z_cols:
        df[col] = df[col] - hip_center_z

    # --- 2. Scaling (using 2D distances) ---
    shoulder_dist_2d = np.linalg.norm(df[left_shoulder_2d].values - df[right_shoulder_2d].values, axis=1)
    hip_dist_2d = np.linalg.norm(df[left_hip_2d].values - df[right_hip_2d].values, axis=1)
    
    scale_factor = (shoulder_dist_2d + hip_dist_2d) / 2.0
    scale_factor[scale_factor == 0] = 1e-6 # Avoid division by zero

    # --- 3. Apply Scaling (The Fix) ---
    # Get all 3D landmark columns for scaling
    landmark_cols = x_cols + y_cols + z_cols

    # Reshape scale_factor to a column vector (e.g., shape (n_frames, 1))
    # This ensures correct broadcasting when dividing the DataFrame
    scale_factor_col = scale_factor.reshape(-1, 1)

    # Divide all centered landmark coordinates by the scale factor
    # This now correctly divides every value in a row by that row's scale factor
    df[landmark_cols] = df[landmark_cols].values / scale_factor_col

    return df

def compare_csv(expert_csv_path, user_csv_path):

    try:
        # Load and preprocess data
        expert_data = load_and_preprocess_data(expert_csv_path)
        user_data = load_and_preprocess_data(user_csv_path)

        # Normalize skeletons
        # expert_data = normalize_skeleton(expert_data)
        # user_data = normalize_skeleton(user_data)

        # Calculate average distance
        results = calculate_detailed_distance(expert_data, user_data)

        # Generate accuracy score (simple linear conversion)
        avg_distance = results["overall_average_distance"]
        accuracy = 1.0 - avg_distance
        accuracy = max(0.0, min(1.0, accuracy))  # Clamp between 0 and 1
        accuracy_percentage = accuracy * 100.0

        # Structure the data for the LLM 
        llm_data = {
            'average_distance': avg_distance,
            'accuracy_score': round(accuracy_percentage, 2),
            'discrepancies': results['discrepancies'],
            'peak_error_timestamp_ms': results['peak_error']['timestamp_ms'],
            'peak_error_magnitude': results['peak_error']['distance']
        }

        print("--- Analysis Complete ---")
        print(f"Accuracy Score: {llm_data['accuracy_score']:.2f}%")
        print("LLM Input Data:")
        print(json.dumps(llm_data, indent=2))

        return llm_data
    
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__": 
    # expert_df = load_and_preprocess_data('backend/assets/front_kick.csv')
    # user_df = load_and_preprocess_data('backend/assets/my_front_kick.csv')
    # print(calculate_detailed_distance(expert_df, user_df, 50))
    results = compare_csv('backend/assets/front_kick.csv', 'backend/assets/my_front_kick.csv')