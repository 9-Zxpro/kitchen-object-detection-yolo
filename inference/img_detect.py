from ultralytics import YOLO
from PIL import Image
import tempfile
import pandas as pd

<<<<<<< HEAD
model = YOLO("../training/weights/best.pt")
=======
model = YOLO("training/weights/best.pt")
>>>>>>> af3349e2879066d9fbad694e76fdd6bdf91cc317

def extract_classes(results):
    names = results[0].names
    classes = results[0].boxes.cls
    confidences = results[0].boxes.conf
    class_names = [names[int(cls)] for cls in classes]
    data = {
        "Class": class_names,
        "Confidence": [f"{conf:.2f}" for conf in confidences],
    }
    return pd.DataFrame(data)

def detect_img(uploaded_file):
    img = Image.open(uploaded_file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        img.save(tmp.name)
        results = model.predict(source=tmp.name, conf=0.5, iou=0.6, verbose=False)

    results[0].save(filename="output.jpg")

    return "output.jpg", extract_classes(results)
