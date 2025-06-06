import argparse
from scripts.train import train
from scripts.predict import predict
from scripts.visualize import visualize_predictions
from scripts.split import split_dataset

def main():
    parser = argparse.ArgumentParser(description="Bowtip detection CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Train the model")

    predict_parser = subparsers.add_parser("predict", help="Run prediction on validation set")
    predict_parser.add_argument("--run", required=False, help="Optional run name")

    visualize_parser = subparsers.add_parser("visualize", help="Visualize predictions with bounding boxes")
    visualize_parser.add_argument("--run", required=False, help="Optional run name")
    visualize_parser.add_argument("--source", default="data/images/val", help="Path to image folder")
    visualize_parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")

    subparsers.add_parser("split", help="Split YOLO dataset into train/val")

    args = parser.parse_args()

    if args.command == "train":
        train()
    elif args.command == "predict":
        predict(run_name=args.run)
    elif args.command == "visualize":
        visualize_predictions(run_name=args.run, source_dir=args.source, conf_threshold=args.conf)
    elif args.command == "split":
        split_dataset()

if __name__ == "__main__":
    main()
