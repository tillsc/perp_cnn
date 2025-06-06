from ultralytics import YOLO
from pathlib import Path

def predict():
    model = YOLO("runs/bugspitzen/weights/best.pt")
    model.predict(
        source="data/images/val",
        save=True,
        save_txt=True,
        save_conf=True
    )

if __name__ == "__main__":
    predict()