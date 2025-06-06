import sys
from scripts.train import train
from scripts.predict import predict
from scripts.split import split_dataset

if __name__ == "__main__":
    if "train" in sys.argv:
        train()
    elif "predict" in sys.argv:
        predict()
    elif "split" in sys.argv:
        split_dataset()
    else:
        print("Usage: python main.py [train|predict|split]")