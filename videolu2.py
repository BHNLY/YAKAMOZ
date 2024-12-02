import cv2
from ultralytics import YOLO

# Modeli yükle
model = YOLO('C:/Users/Oem/PycharmProjects/yolov5/yolov5cam.onnx')

# Video dosyasını aç
video_path = 'C:/Users/Oem/PycharmProjects/yolov5/video.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Kareyi 640x640 boyutuna yeniden boyutlandır
    frame_resized = cv2.resize(frame, (640, 640))

    # Tahmin yap
    results = model(frame_resized)

    # Sonuçları işleyin
    frame = results[0].plot()

    # Görüntüyü göster
    cv2.imshow('YoloV8 Test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
