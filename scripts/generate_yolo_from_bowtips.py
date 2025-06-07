import os
import yaml

DATA_DIR = "./data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LABELS_DIR = os.path.join(DATA_DIR, "labels")
YAML_FILE = "bowtips.yaml"

BOX_WIDTH = 0.10  # normalized width (10%)
BOX_HEIGHT = 0.05  # normalized height (5%)
YOLO_CLASS_ID = 0

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp')

def load_bowtips():
    if os.path.exists(YAML_FILE):
        with open(YAML_FILE, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

def find_images():
    image_files = []
    for root, _, files in os.walk(IMAGES_DIR):
        for file in files:
            if file.lower().endswith(IMAGE_EXTENSIONS):
                image_files.append(os.path.join(root, file))
    return sorted(image_files)

def get_label_path(image_path):
    rel_path = os.path.relpath(image_path, IMAGES_DIR)
    rel_path_txt = os.path.splitext(rel_path)[0] + ".txt"
    return os.path.join(LABELS_DIR, rel_path_txt)

def write_yolo_labels(image_path, bowtips):
    yolo_lines = []
    for norm_x, norm_y in bowtips:
        yolo_line = f"{YOLO_CLASS_ID} {norm_x:.6f} {norm_y:.6f} {BOX_WIDTH:.6f} {BOX_HEIGHT:.6f}"
        yolo_lines.append(yolo_line)

    if yolo_lines:
        label_path = get_label_path(image_path)
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))
        print(f"✅ Wrote labels: {label_path}")

def main():
    bowtip_data = load_bowtips()
    image_files = find_images()

    for image_path in image_files:
        filename = os.path.basename(image_path)
        bowtips = bowtip_data.get(filename)
        if not bowtips:
            continue  # No bowtips for this image
        write_yolo_labels(image_path, bowtips)

    print("🎉 All YOLO labels generated successfully.")

if __name__ == "__main__":
    main()
