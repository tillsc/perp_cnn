from ultralytics import YOLO
from pathlib import Path
import cv2
import argparse
from scripts.utils import get_best_device, get_run_path

def predict_on_image(image_path, model):
    results = model(image_path, conf=0.25)

    for result in results:
        img = result.orig_img.copy()

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{cls} ({conf:.2f})"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(f"Prediction: {Path(image_path).name}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main(source: str, run_name: str = None):
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
    model = YOLO(str(model_path)).to(device)

    source_path = Path(source)
    if source_path.is_file():
        predict_on_image(str(source_path), model)
    elif source_path.is_dir():
        images = list(source_path.glob("*.[jp][pn]g")) + list(source_path.glob("*.webp"))
        if not images:
            print("❌ No images found in directory.")
            return
        for img_path in images:
            predict_on_image(str(img_path), model)
    else:
        print(f"❌ Source not found: {source}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict bounding boxes for external images.")
    parser.add_argument("source", help="Path to image file or folder with images")
    parser.add_argument("--run", help="Name of YOLO run (default: last run)", default=None)
    args = parser.parse_args()

    main(args.source, run_name=args.run)
