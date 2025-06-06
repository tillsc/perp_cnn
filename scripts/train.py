from ultralytics import YOLO

def train():
    model = YOLO("yolov8n-pose.pt")  # oder 's', 'm' etc.
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        project="runs",
        name="bugspitzen",
        exist_ok=True
    )

if __name__ == "__main__":
    train()