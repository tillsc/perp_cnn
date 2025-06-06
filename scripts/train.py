from ultralytics import YOLO
from datetime import datetime

def train():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"bowtips_detect_{timestamp}"

    model = YOLO("yolov8n.pt")  # not yolov8n-pose.pt anymore!
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        project="runs",
        name=run_name,
        exist_ok=True
    )

if __name__ == "__main__":
    train()
