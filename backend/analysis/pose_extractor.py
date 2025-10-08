import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import csv

POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

HEAD_EXCLUDE = {
    'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER',
    'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
    'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT'
}
POSE_SAVE_LANDMARKS = set(lm.name for lm in mp.solutions.pose.PoseLandmark) - HEAD_EXCLUDE

TARGET_FPS = 15
MIN_FRAME_INTERVAL_MS = int(1000 / TARGET_FPS)

POSE_MODEL_PATH = 'backend\models\pose_landmarker_full.task'

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

def analyze_video(video_path, output_csv_path):
    """Analyzes the video to extract video landmarks."""
    if not os.path.exists(POSE_MODEL_PATH):
        print(f"Pose model file not found at {POSE_MODEL_PATH}. Please ensure the model is available.")
        return
    
    base_options = python.BaseOptions
    pose_landmarker_options = vision.PoseLandmarkerOptions(
        base_options=base_options(model_asset_path=POSE_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False
    )

    with vision.PoseLandmarker.create_from_options(pose_landmarker_options) as pose_landmarker, \
         open(output_csv_path, 'w', newline='') as csvfile:
        
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['frame_number', 'timestamp_ms', 'landmark_name', 'visible', 'x', 'y', 'z'])

        cap = cv2.VideoCapture(video_path)

        frame_number = 0
        last_saved_ts = -MIN_FRAME_INTERVAL_MS * 2
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

            # Downsample FPS to save CSV size
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
                                round(float(landmark.visibility), 5),
                                round(float(landmark.x), 5),
                                round(float(landmark.y), 5),
                                round(float(landmark.z), 5)
                            ])

            if should_save:
                last_saved_ts = timestamp_ms

            # For demo visualization
            vis_frame = draw_pose_landmarks(frame.copy(), pose_result)
            cv2.imshow('Analyzer', vis_frame)

            if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
                break

            frame_number += 1

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "backend/assets/my_front_kick.mp4"
    output_csv_path = 'backend/assets/my_front_kick.csv'
    analyze_video(video_path, output_csv_path)
