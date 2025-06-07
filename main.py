import argparse
from scripts.train import train
from scripts.predict import main as predict
from scripts.split import split_dataset
from scripts.label_points import main as run_label_tool
from scripts.generate_yolo_from_bowtips import main as generate_yolo

def main():
    parser = argparse.ArgumentParser(description="Bowtip detection CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--epochs", type=int, default=70, help="Number of training epochs")
    train_parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")

    predict_parser = subparsers.add_parser("predict", help="Run prediction on external image or folder")
    predict_parser.add_argument("source", help="Path to image file or folder of images")
    predict_parser.add_argument("--run", help="YOLO run name (default: latest run)", default=None)

    subparsers.add_parser("split", help="Split YOLO dataset into train/val")

    subparsers.add_parser("label-points", help="Launch interactive tool to manually click bowtips on images")
    subparsers.add_parser("generate-yolo-labels", help="Generate YOLO .txt label files from bowtips.yaml") 

    args = parser.parse_args()

    if args.command == "train":
        train(epochs=args.epochs, patience=args.patience)
    elif args.command == "predict":
        predict(args.source, run_name=args.run)
    elif args.command == "split":
        split_dataset()
    elif args.command == "label-points":
        run_label_tool()
    elif args.command == "generate-yolo-labels":
        generate_yolo()

if __name__ == "__main__":
    main()
