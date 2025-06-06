from ultralytics import YOLO
from datetime import datetime
from scripts.utils import get_best_device

def train(epochs=100, patience=20):
    device = get_best_device()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"bowtips_detect_{timestamp}"

    model = YOLO("yolov8n.pt")
    model.train(
        data="data.yaml",
        epochs=epochs,
        patience=patience,
        imgsz=640,
        device=device,
        name=run_name,
        flipud=0.0,         # ❌ nicht nach oben/unten spiegeln
        fliplr=0.0,         # ❌ nicht horizontal spiegeln (Bug bleibt links)
        degrees=0.0,        # ❌ keine Rotation
        perspective=0.0,    # ❌ keine perspektivische Verzerrung
        shear=0.0,          # ❌ keine Schrägstellung
        translate=0.1,      # ✅ realistische Positionsverschiebung
        scale=0.5,          # ✅ realistische Skalierung
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

if __name__ == "__main__":
    train()
