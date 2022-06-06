import os
import argparse
import shutil
from tqdm import trange
import numpy as np
from utils import get_image_paths


def main(args):
    new_images = get_image_paths(os.path.join(args.data, "new"))
    old_images = get_image_paths(os.path.join(args.data, "old"))
    num_pairs = len(new_images)

    np.random.seed(1234)
    rand_samples = np.random.permutation(num_pairs)

    num_train = int(num_pairs * args.ratio)
    num_test = num_pairs - num_train

    print("# total pairs:", num_pairs)
    print("# train pairs:", num_train)
    print("# test pairs:", num_test)

    train_new_path = os.path.join(args.dest, "train", "new")
    train_old_path = os.path.join(args.dest, "train", "old")
    test_new_path = os.path.join(args.dest, "test", "new")
    test_old_path = os.path.join(args.dest, "test", "old")

    os.makedirs(train_new_path)
    os.makedirs(train_old_path)
    os.makedirs(test_new_path)
    os.makedirs(test_old_path)

    for i in trange(num_pairs):
        if i < num_train:
            dest_new_path = train_new_path
            dest_old_path = train_old_path
        else:
            dest_new_path = test_new_path
            dest_old_path = test_old_path
        rand = rand_samples[i]
        shutil.copy(new_images[rand], dest_new_path)
        shutil.copy(old_images[rand], dest_old_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AmsterTime dataset splitter")
    parser.add_argument("data", help="dataset to be splitted")
    parser.add_argument("--dest", help="destination path")
    parser.add_argument("--ratio", type=float, default=0.7, help="train ratio")
    args = parser.parse_args()

    main(args)
