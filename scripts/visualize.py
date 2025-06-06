from ultralytics import YOLO
from pathlib import Path
import cv2
from scripts.utils import get_best_device, get_run_path

def visualize_predictions(run_name: str = None, source_dir="data/images/val", conf_threshold=0.25):
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

    print(f"✅ Visualizing with model: {model_path}")
    model = YOLO(str(model_path))

    output_dir = Path("predictions/visualized")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(Path(source_dir).glob("*.jpg")) + list(Path(source_dir).glob("*.png")) + list(Path(source_dir).glob("*.webp"))

    for image_path in image_paths:
        results = model.predict(source=str(image_path), save=False, conf=conf_threshold, device=device)
        for r in results:
            img = r.orig_img.copy()
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.imwrite(str(output_dir / image_path.name), img)

    print(f"✅ Saved {len(image_paths)} annotated images to: {output_dir.resolve()}")
