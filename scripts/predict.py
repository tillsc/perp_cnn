from ultralytics import YOLO
from pathlib import Path

def predict():
    runs_path = Path("runs/detect")
    run_dirs = sorted(runs_path.glob("bowtips_detect_*"), key=lambda d: d.stat().st_mtime, reverse=True)

    if not run_dirs:
        print("❌ No trained detection models found.")
        return

    latest_run = run_dirs[0]
    model_path = latest_run / "weights" / "best.pt"

    if not model_path.exists():
        print(f"❌ No weights found at: {model_path}")
        return

    print(f"✅ Using model: {model_path}")

    model = YOLO(str(model_path))
    model.predict(
        source="data/images/val",
        save=True,
        save_txt=True,
        save_conf=True
    )

if __name__ == "__main__":
    predict()
