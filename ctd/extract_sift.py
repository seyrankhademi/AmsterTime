import cv2 as cv
import numpy as np
import os
import argparse
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import get_image_paths, get_path_info, classes


def extract_sift(filepath, sift, draw=False):
    img = cv.imread(filepath)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kp, descs = sift.detectAndCompute(gray, None)
    if draw:
        img = cv.drawKeypoints(
            gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return img
    return kp, descs


def main(args):
    sift = cv.SIFT_create()
    if args.single_file:
        new_filepath = os.path.join(args.data, 'new', args.single_file)
        new_img = extract_sift(new_filepath, sift, draw=True)
        old_filepath = os.path.join(args.data, 'old', args.single_file)
        old_img = extract_sift(old_filepath, sift, draw=True)
        img = np.hstack((new_img, old_img))
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        plt.show()
        return

    descriptors = {}
    if os.path.exists(args.file):
        with open(args.file, 'rb') as f:
            descriptors = pickle.load(f)

    num_descriptors = 0
    for cname in classes:
        cdescs = descriptors.get(cname, {})
        image_paths = get_image_paths(os.path.join(args.data, cname))
        for path in tqdm(image_paths, total=len(image_paths)):
            fname = get_path_info(path)[2]
            if fname not in cdescs:
                _, descs = extract_sift(path, sift)
                if descs is None:
                    print(path, ' has no sift descsiptors')
                    continue
                num_descriptors += len(descs)
                cdescs[fname] = descs
        descriptors[cname] = cdescs

    print('# of extracted sift descriptors:', num_descriptors)
    with open(args.file, 'wb') as f:
        pickle.dump(descriptors, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SIFT feature extractor')
    parser.add_argument('data', help='path to data set')
    parser.add_argument('--file', default='sift_descs.p',
                        help='path to pickle file to save sift descriptors')
    parser.add_argument('--single-file', required=False,
                        help='file to visualize sift features')
    args = parser.parse_args()

    main(args)
