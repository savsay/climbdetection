import cv2
from ultralytics import YOLO
import torch
import os
import numpy as np

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def main():
    torch.cuda.is_available = lambda: False
    print("Loading YOLOv8 pose model...")
    model_path = 'yolov8n-pose.pt'
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Downloading...")
        model = YOLO('yolov8n-pose.pt')
    else:
        model = YOLO(model_path)
    video_path = 'input_video.mp4'
    print(f"Opening video file: {video_path}")
    if not os.path.exists(video_path):
        raise Exception(f"Video file not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Error: Could not open video file {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"Video properties: {width}x{height} @ {fps}fps")
    # Yatay mı dikey mi kontrol et
    is_portrait = width < height
    if is_portrait:
        print("Portrait video detected. Rotating frames 90 degrees for processing.")
        output_width, output_height = height, width
    else:
        output_width, output_height = width, height
    output_path = 'output_pose_v5.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    if not out.isOpened():
        raise Exception(f"Error: Could not create output video file {output_path}")
    print("Starting video processing...")
    hip_points = []
    prev_hip = None
    MAX_DIST = 100
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Eğer dikeyse frame'i döndür
        if is_portrait:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        results = model(frame, conf=0.3)
        hip_candidates = []
        keypoint_datas = []
        for result in results:
            keypoints = result.keypoints
            boxes = result.boxes
            for i, box in enumerate(boxes):
                if int(box.cls[0]) == 0 and keypoints is not None:
                    keypoint_data = keypoints[i].data[0]
                    left_hip = keypoint_data[11][:2]
                    right_hip = keypoint_data[12][:2]
                    hip_x = int((left_hip[0] + right_hip[0]) / 2)
                    hip_y = int((left_hip[1] + right_hip[1]) / 2)
                    if hip_y > int(0.2 * frame.shape[0]):
                        hip_candidates.append((hip_x, hip_y))
                        keypoint_datas.append(keypoint_data)
        # En yakın hip noktasını ve ona ait keypoint datasını seç
        hip_point = None
        selected_keypoint_data = None
        if hip_candidates:
            if prev_hip is None:
                hip_point = hip_candidates[0]
                selected_keypoint_data = keypoint_datas[0]
            else:
                min_idx = np.argmin([euclidean_distance(p, prev_hip) for p in hip_candidates])
                if euclidean_distance(hip_candidates[min_idx], prev_hip) <= MAX_DIST:
                    hip_point = hip_candidates[min_idx]
                    selected_keypoint_data = keypoint_datas[min_idx]
        if hip_point is not None:
            hip_points.append(hip_point)
            prev_hip = hip_point
        # Çizim: yol ve hip noktaları
        for i in range(1, len(hip_points)):
            cv2.line(frame, hip_points[i-1], hip_points[i], (0, 255, 255), 2)
        for pt in hip_points:
            cv2.circle(frame, pt, 4, (0, 255, 255), -1)
        if len(hip_points) > 0:
            cv2.circle(frame, hip_points[-1], 8, (0, 255, 255), -1)
        # Çizim: sadece takip edilen kişinin iskeleti
        if selected_keypoint_data is not None:
            for kp in selected_keypoint_data:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
            connections = [
                (5, 7), (7, 9), (6, 8), (8, 10),
                (5, 6), (11, 12),
                (11, 13), (13, 15), (12, 14), (14, 16),
                (5, 11), (6, 12)
            ]
            for c in connections:
                x1, y1 = int(selected_keypoint_data[c[0]][0]), int(selected_keypoint_data[c[0]][1])
                x2, y2 = int(selected_keypoint_data[c[1]][0]), int(selected_keypoint_data[c[1]][1])
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        out.write(frame)
        cv2.imshow('Pose Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == '__main__':
    main()
