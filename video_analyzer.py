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

def draw_pose_landmarks(image, detection_result):
    """Draws the pose landmarks and connections on the image."""
    if detection_result.pose_landmarks:
        for landmarks in detection_result.pose_landmarks:
            for i in range(len(landmarks)):
                x = int(landmarks[i].x * image.shape[1])
                y = int(landmarks[i].y * image.shape[0])
                cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
            # Draw pose connections
            for connection in POSE_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                if landmarks[start_idx].visibility > 0.5 and landmarks[end_idx].visibility > 0.5:
                    start_point = (int(landmarks[start_idx].x * image.shape[1]), int(landmarks[start_idx].y * image.shape[0]))
                    end_point = (int(landmarks[end_idx].x * image.shape[1]), int(landmarks[end_idx].y * image.shape[0]))
                    cv2.line(image, start_point, end_point, (0, 255, 0), 2)
    return image

def draw_hand_landmarks(image, detection_result):
    """Draws the hand landmarks and connections on the image."""
    if detection_result.hand_landmarks:
        for landmarks in detection_result.hand_landmarks:
            for i in range(len(landmarks)):
                x = int(landmarks[i].x * image.shape[1])
                y = int(landmarks[i].y * image.shape[0])
                cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
            # Draw hand connections
            for connection in HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
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
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

            if pose_result.pose_landmarks:
                for i, landmark_list in enumerate(pose_result.pose_landmarks):
                    for j, landmark in enumerate(landmark_list):
                        landmark_name = f'pose_{mp.solutions.pose.PoseLandmark(j).name}'
                        csv_writer.writerow([frame_number, timestamp_ms, landmark_name, landmark.x, landmark.y, landmark.z])
            
            if hand_result.hand_landmarks:
                for i, landmark_list in enumerate(hand_result.hand_landmarks):
                    handedness = hand_result.handedness[i][0].category_name
                    for j, landmark in enumerate(landmark_list):
                        landmark_name = f'{handedness}_hand_{mp.solutions.hands.HandLandmark(j).name}'
                        csv_writer.writerow([frame_number, timestamp_ms, landmark_name, landmark.x, landmark.y, landmark.z])

            # For visualization
            annotated_image = draw_pose_landmarks(frame.copy(), pose_result)
            annotated_image = draw_hand_landmarks(annotated_image, hand_result)
            cv2.imshow('MediaPipe Landmarks', annotated_image)

            if cv2.waitKey(5) & 0xFF == 27: # Press ESC to exit
                break
            
            frame_number += 1

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
