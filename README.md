# Perp CNN – Training Setup

This repository contains a training setup for detecting boat bow tips in photofinish images using YOLOv8 pose estimation.

## 🧩 Dataset Preparation

### 1. Export image data and labels

From the `perp_web` Rails app, run the following Rake task:

```
bundle exec rake export:cnn_raw_data
```

This will download finish line images and generate normalized `.txt` label files (YOLOv8 pose format) under:

```
tmp/cnn_data/
├── images/
└── labels/
```

### 2. Copy exported data to this project

From the Rails project directory:

```
cp -r tmp/cnn_data/* data/
```

### 3. Split into train/val sets

From the CNN project root directory:

```
python main.py split
```

This creates the following structure:

```
data/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

## 🚀 Training

Run YOLOv8 training:

```
python main.py train
```

This will start training a keypoint model to detect the bow tip of the boat using YOLOv8 pose estimation.

## 🔎 Inference

After training, run prediction on the validation images:

```
python main.py predict
```

Predicted images and `.txt` outputs will be saved in `runs/predict/`.

## 🧪 Notes

- Labels are generated in YOLOv8 **pose format**:
  ```
  class x y visibility
  ```
  Example:
  ```
  0 0.345678 0.543210 2
  ```

- Only one keypoint (the bowtip) is used per object.
- The x-value is based on the finish timestamp.
- The y-value is estimated as the vertical center of the boat's lane.

---

**Happy training!** 🚣‍♀️
