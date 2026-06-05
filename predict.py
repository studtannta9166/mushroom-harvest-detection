from ultralytics import YOLO

# load your trained model
model = YOLO('best.pt')

# test on one image from your dataset
results = model('datasets/merged/test/images/', save=True)

print("done! check runs/detect/predict/ for results")




from ultralytics import YOLO

model = YOLO('best.pt')

# photo you want to predict on
results = model('/Users/tanu/Downloads/my_mushroom.jpg', save=True)

# print what it detected
for result in results:
    for box in result.boxes:
        class_id = int(box.cls)
        confidence = float(box.conf)
        class_name = model.names[class_id]
        print(f"Detected: {class_name} — confidence: {confidence:.2%}")