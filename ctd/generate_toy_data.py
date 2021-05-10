import cv2 as cv
import numpy as np
import os
import argparse
from tqdm import tqdm
from .utils import get_image_paths, get_path_info


def main(args):
    os.makedirs(os.path.join(args.toy_data, 'new'), exist_ok=True)
    os.makedirs(os.path.join(args.toy_data, 'old'), exist_ok=True)

    paths = get_image_paths(args.data, 'new')[:100]
    for path in tqdm(paths, total=len(paths)):
        new_img = cv.imread(path).astype(np.float)
        old_img = np.clip(new_img * 1.5 + 30, 0, 255)
        _, _, fname = get_path_info(path)
        new_path = os.path.join(args.toy_data, 'new', fname)
        old_path = os.path.join(args.toy_data, 'old', fname)
        cv.imwrite(new_path, new_img)
        cv.imwrite(old_path, old_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Toy data generator')
    parser.add_argument('data', help='path to data set')
    parser.add_argument('--toy-data', default='toy_data',
                        help='path to toy data set')
    args = parser.parse_args()

    main(args)
