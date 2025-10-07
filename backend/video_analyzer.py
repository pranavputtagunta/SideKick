# Clean imports
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import argparse
import csv
import time
import os

POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

# Record all pose landmarks except a small head-exclusion set to reduce CSV size
# We'll exclude eyes, ears, nose and mouth-related landmarks but keep the full body
HEAD_EXCLUDE = {
    'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER',
    'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
    'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT'
}
# Compute the set of pose landmarks to save (all minus head landmarks)
POSE_SAVE_LANDMARKS = set(lm.name for lm in mp.solutions.pose.PoseLandmark) - HEAD_EXCLUDE
HAND_SAVE_LANDMARKS = {
    'WRIST', 'THUMB_TIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_TIP',
    'RING_FINGER_TIP', 'PINKY_TIP', 'INDEX_FINGER_MCP', 'MIDDLE_FINGER_MCP'
}

# Target recording frame rate (Hz)
TARGET_FPS = 15
MIN_FRAME_INTERVAL_MS = int(1000 / TARGET_FPS)


def draw_pose_landmarks(image, detection_result):
    """Draws the pose landmarks and connections on the image."""
    if getattr(detection_result, 'pose_landmarks', None):
        for landmarks in detection_result.pose_landmarks:
            # Draw ALL pose landmarks for visualization
            for i in range(len(landmarks)):
                x = int(landmarks[i].x * image.shape[1])
                y = int(landmarks[i].y * image.shape[0])
                cv2.circle(image, (x, y), 3, (200, 200, 50), -1)
            # Draw all pose connections
            for connection in POSE_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                # check visibility when available
                try:
                    if landmarks[start_idx].visibility > 0.5 and landmarks[end_idx].visibility > 0.5:
                        start_point = (int(landmarks[start_idx].x * image.shape[1]), int(landmarks[start_idx].y * image.shape[0]))
                        end_point = (int(landmarks[end_idx].x * image.shape[1]), int(landmarks[end_idx].y * image.shape[0]))
                        cv2.line(image, start_point, end_point, (200, 200, 50), 2)
                except Exception:
                    # If no visibility field, draw the connection anyway
                    start_point = (int(landmarks[start_idx].x * image.shape[1]), int(landmarks[start_idx].y * image.shape[0]))
                    end_point = (int(landmarks[end_idx].x * image.shape[1]), int(landmarks[end_idx].y * image.shape[0]))
                    cv2.line(image, start_point, end_point, (200, 200, 50), 2)
    return image


def draw_hand_landmarks(image, detection_result):
    """Draws the hand landmarks and connections on the image."""
    if getattr(detection_result, 'hand_landmarks', None):
        for landmarks in detection_result.hand_landmarks:
            recorded_indices = set()
            for i in range(len(landmarks)):
                lm_name = mp.solutions.hands.HandLandmark(i).name
                if lm_name in HAND_SAVE_LANDMARKS:
                    x = int(landmarks[i].x * image.shape[1])
                    y = int(landmarks[i].y * image.shape[0])
                    cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
                    recorded_indices.add(i)
            # Draw hand connections only between recorded landmarks
            for connection in HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                if start_idx in recorded_indices and end_idx in recorded_indices:
                    start_point = (int(landmarks[start_idx].x * image.shape[1]), int(landmarks[start_idx].y * image.shape[0]))
                    end_point = (int(landmarks[end_idx].x * image.shape[1]), int(landmarks[end_idx].y * image.shape[0]))
                    cv2.line(image, start_point, end_point, (0, 0, 255), 2)
    return image


def main():
    parser = argparse.ArgumentParser(description='Video landmark detection using MediaPipe.')
    parser.add_argument('--video_file', type=str, default=None, help='Path to the video file.')
    parser.add_argument('--camera_id', type=int, default=0, help='ID of the camera to use.')
    parser.add_argument('--output_csv', type=str, default='landmarks.csv', help='Path to the output CSV file.')
    args = parser.parse_args()

    hand_model_path = 'models/hand_landmarker.task'
    pose_model_path = 'models/pose_landmarker_full.task'

    if not os.path.exists(hand_model_path):
        print(f"Hand landmarker model not found at {hand_model_path}")
        return
    if not os.path.exists(pose_model_path):
        print(f"Pose landmarker model not found at {pose_model_path}")
        return

    base_options = python.BaseOptions
    hand_landmarker_options = vision.HandLandmarkerOptions(
        base_options=base_options(model_asset_path=hand_model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2)

    pose_landmarker_options = vision.PoseLandmarkerOptions(
        base_options=base_options(model_asset_path=pose_model_path),
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False)

    with vision.HandLandmarker.create_from_options(hand_landmarker_options) as hand_landmarker, \
         vision.PoseLandmarker.create_from_options(pose_landmarker_options) as pose_landmarker, \
         open(args.output_csv, 'w', newline='') as csvfile:

        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['frame_number', 'timestamp_ms', 'landmark_name', 'x', 'y', 'z'])

        if args.video_file:
            cap = cv2.VideoCapture(args.video_file)
        else:
            cap = cv2.VideoCapture(args.camera_id)

        frame_number = 0
        last_saved_ts = -MIN_FRAME_INTERVAL_MS * 2
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

            # Downsample to target FPS when writing to CSV
            should_save = (timestamp_ms - last_saved_ts) >= MIN_FRAME_INTERVAL_MS

            if should_save and getattr(pose_result, 'pose_landmarks', None):
                for i, landmark_list in enumerate(pose_result.pose_landmarks):
                    for j, landmark in enumerate(landmark_list):
                        lm_name = mp.solutions.pose.PoseLandmark(j).name
                        if lm_name in POSE_SAVE_LANDMARKS:
                            landmark_name = f'pose_{lm_name}'
                            csv_writer.writerow([
                                frame_number,
                                timestamp_ms,
                                landmark_name,
                                round(float(landmark.x), 5),
                                round(float(landmark.y), 5),
                                round(float(landmark.z), 5),
                            ])

            if should_save and getattr(hand_result, 'hand_landmarks', None):
                for i, landmark_list in enumerate(hand_result.hand_landmarks):
                    # handedness may be e.g. 'Left' or 'Right'
                    try:
                        handedness = hand_result.handedness[i][0].category_name
                    except Exception:
                        handedness = 'Unknown'
                    for j, landmark in enumerate(landmark_list):
                        lm_name = mp.solutions.hands.HandLandmark(j).name
                        if lm_name in HAND_SAVE_LANDMARKS:
                            landmark_name = f'{handedness}_hand_{lm_name}'
                            csv_writer.writerow([
                                frame_number,
                                timestamp_ms,
                                landmark_name,
                                round(float(landmark.x), 5),
                                round(float(landmark.y), 5),
                                round(float(landmark.z), 5),
                            ])

            if should_save:
                last_saved_ts = timestamp_ms

            # For visualization
            annotated = draw_pose_landmarks(frame.copy(), pose_result)
            annotated = draw_hand_landmarks(annotated, hand_result)
            cv2.imshow('MediaPipe Landmarks', annotated)

            if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
                break

            frame_number += 1

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
