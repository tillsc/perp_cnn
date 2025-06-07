import argparse
from scripts.train import train
from scripts.predict import predict
from scripts.visualize import visualize_predictions
from scripts.split import split_dataset
from scripts.label_points import main as run_label_tool
from scripts.generate_yolo_from_bowtips import main as generate_yolo

def main():
    parser = argparse.ArgumentParser(description="Bowtip detection CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--epochs", type=int, default=70, help="Number of training epochs")
    train_parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")

    predict_parser = subparsers.add_parser("predict", help="Run prediction on validation set")
    predict_parser.add_argument("--run", required=False, help="Optional run name")

    visualize_parser = subparsers.add_parser("visualize", help="Visualize predictions with bounding boxes")
    visualize_parser.add_argument("--run", required=False, help="Optional run name")
    visualize_parser.add_argument("--source", default="data/images/val", help="Path to image folder")
    visualize_parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")

    subparsers.add_parser("split", help="Split YOLO dataset into train/val")

    subparsers.add_parser("label-points", help="Launch interactive tool to manually click bowtips on images")
    subparsers.add_parser("generate-yolo-labels", help="Generate YOLO .txt label files from bowtips.yaml") 

    args = parser.parse_args()

    if args.command == "train":
        train(epochs=args.epochs, patience=args.patience)
    elif args.command == "predict":
        predict(run_name=args.run)
    elif args.command == "visualize":
        visualize_predictions(run_name=args.run, source_dir=args.source, conf_threshold=args.conf)
    elif args.command == "split":
        split_dataset()
    elif args.command == "label-points":
        run_label_tool()
    elif args.command == "generate-yolo-labels":
        generate_yolo()

if __name__ == "__main__":
    main()
