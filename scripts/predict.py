from ultralytics import YOLO
from pathlib import Path
from scripts.utils import get_best_device, get_run_path

def predict(run_name: str = None):
    device = get_best_device()
    try:
        run_dir = get_run_path(run_name)
    except FileNotFoundError as e:
        print(e)
        return

    model_path = run_dir / "weights" / "best.pt"
    if not model_path.exists():
        print(f"❌ No weights found at: {model_path}")
        return

    print(f"✅ Using model: {model_path}")
    model = YOLO(str(model_path))

    model.predict(
        source="data/images/val",
        save=True,
        conf=0.25,
        device=device
    )

if __name__ == "__main__":
    predict()
