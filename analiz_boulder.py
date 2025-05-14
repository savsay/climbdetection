import cv2
from ultralytics import YOLO
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def calibrate_scale(frame, reference_height_meters=4.0):
    """
    Kullanıcıdan referans noktaları alarak piksel-metre dönüşüm oranını hesaplar
    """
    print("\nKalibrasyon için duvarın üst ve alt noktalarını seçin:")
    print("1. İlk tıklama: Duvarın alt noktası")
    print("2. İkinci tıklama: Duvarın üst noktası")
    
    points = []
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Kalibrasyon', frame)
            if len(points) == 2:
                cv2.destroyAllWindows()
    
    cv2.imshow('Kalibrasyon', frame)
    cv2.setMouseCallback('Kalibrasyon', click_event)
    cv2.waitKey(0)
    
    if len(points) != 2:
        raise Exception("Kalibrasyon için iki nokta seçilmedi!")
    
    # Piksel cinsinden yükseklik
    pixel_height = abs(points[1][1] - points[0][1])
    
    # Piksel/metre oranı
    pixels_per_meter = pixel_height / reference_height_meters
    
    return pixels_per_meter

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
    
    # İlk frame'i al ve kalibrasyon yap
    ret, first_frame = cap.read()
    if not ret:
        raise Exception("Could not read first frame for calibration")
    
    pixels_per_meter = calibrate_scale(first_frame)
    print(f"Kalibrasyon tamamlandı. 1 metre = {pixels_per_meter:.2f} piksel")
    
    # Video işleme
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"Video properties: {width}x{height} @ {fps}fps")
    output_path = 'output_pose_v5.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise Exception(f"Error: Could not create output video file")
    print("Starting video processing...")
    hip_points = []
    prev_hip = None
    MAX_DIST = 100
    MIN_DIST = 15  # Minimum mesafe eşiği (piksel)
    frame_idx = 0
    heights = []  # Her frame için hip_y (tırmanış yüksekliği)
    speeds = []   # Her frame için hız (px/frame)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
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
        
        # Yeni hip noktası eklenecek mi kontrol et
        if hip_point is not None:
            if prev_hip is None or euclidean_distance(hip_point, prev_hip) >= MIN_DIST:
                hip_points.append(hip_point)
                prev_hip = hip_point
                heights.append(hip_point[1])
                if len(hip_points) > 1:
                    speeds.append(euclidean_distance(hip_points[-1], hip_points[-2]) * fps)
                else:
                    speeds.append(0)
            else:
                heights.append(heights[-1] if heights else np.nan)
                speeds.append(0)
        else:
            heights.append(np.nan)
            speeds.append(0)
        # Normal video çizimi
        frame_normal = frame.copy()
        for i in range(1, len(hip_points)):
            cv2.line(frame_normal, hip_points[i-1], hip_points[i], (0, 255, 255), 2)
        for pt in hip_points:
            cv2.circle(frame_normal, pt, 4, (0, 255, 255), -1)
        if len(hip_points) > 0:
            cv2.circle(frame_normal, hip_points[-1], 8, (0, 255, 255), -1)
        # Başlangıç ve bitiş noktası
        if len(hip_points) > 0:
            cv2.circle(frame_normal, hip_points[0], 10, (0, 0, 255), 2)  # Başlangıç kırmızı halka
            cv2.circle(frame_normal, hip_points[-1], 10, (0, 255, 0), 2)  # Bitiş yeşil halka
        if selected_keypoint_data is not None:
            for kp in selected_keypoint_data:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(frame_normal, (x, y), 4, (0, 0, 255), -1)
            connections = [
                (5, 7), (7, 9), (6, 8), (8, 10),
                (5, 6), (11, 12),
                (11, 13), (13, 15), (12, 14), (14, 16),
                (5, 11), (6, 12)
            ]
            for c in connections:
                x1, y1 = int(selected_keypoint_data[c[0]][0]), int(selected_keypoint_data[c[0]][1])
                x2, y2 = int(selected_keypoint_data[c[1]][0]), int(selected_keypoint_data[c[1]][1])
                cv2.line(frame_normal, (x1, y1), (x2, y2), (255, 0, 0), 2)
        out.write(frame_normal)
        cv2.imshow('Normal Tracking', frame_normal)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_idx += 1
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Done.")

    # --- PERFORMANS ANALİZİ ---
    # Tüm analizleri tek bir figure'da göster
    plt.figure(figsize=(15, 10))
    
    # 1. Yükseklik grafiği (metre cinsinden)
    plt.subplot(2, 2, 1)
    heights_meters = [h/pixels_per_meter for h in heights]
    plt.plot(np.arange(len(heights_meters))/fps, heights_meters, 'b-', label='Yükseklik')
    plt.xlabel('Zaman (s)')
    plt.ylabel('Yükseklik (m)')
    plt.title('Tırmanış Yüksekliği Zaman Grafiği')
    plt.grid(True)
    
    # 2. Hız grafiği (m/s cinsinden)
    plt.subplot(2, 2, 2)
    speeds_ms = [s/pixels_per_meter for s in speeds]
    plt.plot(np.arange(len(speeds_ms))/fps, speeds_ms, 'r-', label='Hız')
    plt.xlabel('Zaman (s)')
    plt.ylabel('Hız (m/s)')
    plt.title('Tırmanış Hızı Zaman Grafiği')
    plt.grid(True)
    
    # 3. Metrikler
    plt.subplot(2, 2, 3)
    plt.axis('off')
    
    # Toplam yol uzunluğunu hesapla (sadece anlamlı hareketler)
    total_path = 0
    for i in range(1, len(hip_points)):
        dist = euclidean_distance(hip_points[i], hip_points[i-1])
        if dist >= MIN_DIST:  # Sadece minimum mesafeden büyük hareketleri topla
            total_path += dist
    
    total_path_meters = total_path / pixels_per_meter
    climb_duration = len(hip_points) / fps
    avg_speed = total_path_meters / climb_duration
    
    metrics_text = f"""
    PERFORMANS METRİKLERİ:
    
    Toplam Yol Uzunluğu: {total_path_meters:.2f} metre
    Tırmanış Süresi: {climb_duration:.2f} saniye
    Ortalama Hız: {avg_speed:.2f} m/s
    
    Başlangıç Noktası: {hip_points[0] if len(hip_points) > 0 else 'N/A'}
    Bitiş Noktası: {hip_points[-1] if len(hip_points) > 0 else 'N/A'}
    """
    plt.text(0.1, 0.5, metrics_text, fontsize=12, va='center')
    
    # 4. Yol çizimi (metre cinsinden)
    plt.subplot(2, 2, 4)
    if len(hip_points) > 0:
        x_coords = [p[0]/pixels_per_meter for p in hip_points]
        y_coords = [p[1]/pixels_per_meter for p in hip_points]
        plt.plot(x_coords, y_coords, 'g-', label='Tırmanış Yolu')
        plt.scatter(x_coords[0], y_coords[0], color='red', s=100, label='Başlangıç')
        plt.scatter(x_coords[-1], y_coords[-1], color='green', s=100, label='Bitiş')
    plt.title('Tırmanış Yolu Görselleştirmesi')
    plt.xlabel('Yatay Mesafe (m)')
    plt.ylabel('Dikey Mesafe (m)')
    plt.legend()
    plt.grid(True)
    
    # Grafikleri kaydet
    plt.tight_layout()
    plt.savefig('boulder_analysis_summary.png', dpi=300, bbox_inches='tight')
    print("\nTüm analizler 'boulder_analysis_summary.png' dosyasına kaydedildi.")
    
    # Açıklamalar
    print("\n--- Açıklamalar ---")
    print("- Toplam yol uzunluğu: Tırmanıcının katettiği toplam mesafe (metre cinsinden).")
    print("- Yükseklik grafiği: Tırmanıcının zaman içindeki dikey hareketi (metre cinsinden).")
    print("- Hız grafiği: Tırmanıcının anlık hız değişimi (metre/saniye cinsinden).")
    print("- Tırmanış yolu: Tırmanıcının izlediği yolun 2D görselleştirmesi (metre cinsinden).")
    print("- Tırmanış süresi: Toplam tırmanış süresi (saniye cinsinden).")

if __name__ == '__main__':
    main() 