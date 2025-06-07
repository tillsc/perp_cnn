import cv2
import os
import yaml
import numpy as np

DATA_DIR = "./data"
YAML_FILE = "bowtips.yaml"
WINDOW_NAME = "Bowtip Labeling"

# Key codes for arrow navigation
KEY_LEFT = 63234
KEY_RIGHT = 63235

RADIUS = 5
POINT_COLOR = (0, 0, 255)  # Red in BGR

def find_all_images():
    extensions = ('.png', '.jpg', '.jpeg', '.webp')
    image_files = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.lower().endswith(extensions):
                image_files.append(os.path.join(root, file))
    return sorted(image_files)

def load_yaml():
    if os.path.exists(YAML_FILE):
        with open(YAML_FILE, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}

def save_yaml(data):
    with open(YAML_FILE, 'w') as f:
        yaml.dump(data, f)

def denormalize_point(x_norm, y_norm, width, height):
    x = int(x_norm * width)
    y = int(y_norm * height)
    return [x, y]

def normalize_point(x, y, width, height):
    return [x / width, y / height]

def draw_points(img, norm_points, width, height):
    for x_norm, y_norm in norm_points:
        x, y = denormalize_point(x_norm, y_norm, width, height)

        # Draw red dot at the center
        cv2.circle(img, (x, y), RADIUS, POINT_COLOR, -1)

        # Calculate box size
        box_w = int(width * 0.1)
        box_h = int(height * 0.05)

        # Top-left and bottom-right corners
        x1 = max(0, x - box_w // 2)
        y1 = max(0, y - box_h // 2)
        x2 = min(width - 1, x + box_w // 2)
        y2 = min(height - 1, y + box_h // 2)

        # Draw thin green rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)

def main():
    images = find_all_images()
    if not images:
        print("❌ No images found.")
        return

    data = load_yaml()
    index = 0
    current_points = []

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        image_path = images[index]
        filename = os.path.basename(image_path)
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ Could not load image: {image_path}")
            index = (index + 1) % len(images)
            continue

        height, width = img.shape[:2]
        current_points = data.get(filename, []).copy()

        def click_event(event, x, y, flags, param):
            nonlocal current_points
            if event == cv2.EVENT_LBUTTONDOWN:
                norm_x, norm_y = normalize_point(x, y, width, height)
                current_points.append([norm_x, norm_y])
                data[filename] = current_points
                save_yaml(data)

        cv2.setMouseCallback(WINDOW_NAME, click_event)

        while True:
            display = img.copy()
            draw_points(display, current_points, width, height)
            cv2.putText(display, f"{os.path.relpath(image_path, DATA_DIR)} ({index+1}/{len(images)})",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKeyEx(20)

            if key == 27 or key == ord('q'): 
                cv2.destroyAllWindows()
                save_yaml(data)
                print("👋 Exiting and saving.")
                return
            elif key == ord('r'):
                current_points = []
                data[filename] = current_points
                save_yaml(data)
                print(f"🔄 Reset points for {filename}")
            elif key == KEY_RIGHT or key == ord('d'):
                index = (index + 1) % len(images)
                break
            elif key == KEY_LEFT or key == ord('a'):
                index = (index - 1) % len(images)
                break

if __name__ == "__main__":
    main()
