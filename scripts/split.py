import os
import random
import shutil
from pathlib import Path

def split_dataset():
    BASE = Path("data")
    IMAGES = BASE / "images"
    LABELS = BASE / "labels"

    for subset in ["train", "val"]:
        (IMAGES / subset).mkdir(parents=True, exist_ok=True)
        (LABELS / subset).mkdir(parents=True, exist_ok=True)

    image_files = list(IMAGES.glob("*.webp"))
    random.shuffle(image_files)

    split_ratio = 0.8
    split_idx = int(len(image_files) * split_ratio)

    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    def move(files, subset):
        for img_path in files:
            label_path = LABELS / img_path.with_suffix(".txt").name
            if not label_path.exists():
                continue
            shutil.move(str(img_path), IMAGES / subset / img_path.name)
            shutil.move(str(label_path), LABELS / subset / label_path.name)

    move(train_files, "train")
    move(val_files, "val")

    print(f"Split complete: {len(train_files)} train / {len(val_files)} val")
