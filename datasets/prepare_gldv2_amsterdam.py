import argparse
import os
import shutil
import pandas as pd
from tqdm import tqdm


def prepare_csv(train_clean_path):
    clean = pd.read_csv(train_clean_path)
    selected_path = os.path.join(os.path.dirname(__file__), "selected_landmarks.csv")
    amsterdam = pd.read_csv(selected_path)
    clean = clean.merge(amsterdam, on="landmark_id").filter(
        items=["landmark_id", "images"]
    )
    return clean


def create_dataset(root, root_amsterdam, df):
    if not os.path.exists(root_amsterdam):
        os.makedirs(root_amsterdam)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        landmark_id = row["landmark_id"]
        landmark_path = os.path.join(root_amsterdam, str(landmark_id))
        if not os.path.exists(landmark_path):
            os.makedirs(landmark_path)

        images = row["images"].split()
        for i in images:
            img_path = os.path.expanduser(
                os.path.join(root, "train", *i[:3], f"{i}.jpg")
            )
            if os.path.exists(img_path):
                shutil.copy(img_path, os.path.join(landmark_path, f"{i}.jpg"))
            else:
                print(f"Image not found: {img_path}")


def main(args):
    root = os.path.expanduser(args.root)
    train_clean_path = os.path.join(root, "train/train_clean.csv")
    clean = prepare_csv(train_clean_path)
    amsterdam_path = os.path.join(
        os.path.dirname(__file__), "gldv2_clean_amsterdam/train"
    )
    create_dataset(root, amsterdam_path, clean)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLDv2 Amsterdam preperation")
    parser.add_argument("root", help="GLDv2 dataset root path")
    args = parser.parse_args()
    main(args)
