from ultralytics import YOLO

model = YOLO("yolov8s.pt")  # small, not nano — better accuracy for real world

model.train(
    data="datasets/merged/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="mushroom_detector_v1",
    patience=10,        # stops early if no improvement for 10 epochs
    augment=True,       # random flips, brightness changes — helps generalize
    hsv_h=0.015,        # slight hue variation — handles different lighting
    hsv_v=0.4,          # brightness variation — important for dim farm rooms
    fliplr=0.5,         # horizontal flip
    mosaic=1.0,         # combines 4 images — helps detect small mushrooms
)