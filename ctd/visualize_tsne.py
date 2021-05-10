import os
import sys
import argparse
import pickle
import numpy as np
import cv2 as cv
from utils import reduce_dim

# To create tsne plot, download and compile https://github.com/lvdmaaten/bhtsne
# in the parent directory.
sys.path.append('../bhtsne/')
try:
    import bhtsne
except Exception as e:
    print(e)
    print('Couldn\'t import bhtsne. it must be in your path.')
    exit()

classes = ['new', 'old']
class_colors = {'new': (255, 0, 0), 'old': (0, 255, 0)}
tiny_size = 128
img_size = 4000


def load_img(img_path):
    img = cv.imread(img_path)
    height, width = img.shape[:2]
    long_edge = max(height, width)
    rh, rw = height / long_edge, width / long_edge
    height = int(rh * tiny_size)
    width = int(rw * tiny_size)
    return cv.resize(img, (width, height),
                     interpolation=cv.INTER_CUBIC)


def main(args):
    descs_path = args.descs
    if not os.path.exists(descs_path):
        print('File containing descs couldn\'t found')
        exit()

    with open(descs_path, 'rb') as f:
        descriptors = pickle.load(f)

    X = []
    imgs = []
    for cname in descriptors:
        for fname in descriptors[cname]:
            descs = descriptors[cname][fname]
            X.append(descs)
            img_path = os.path.join(args.data, cname, fname)
            img = load_img(img_path)
            cv.putText(img, fname, (0, 7),
                       cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, class_colors[cname], 1)
            imgs.append(img)
    X = np.array(X)
    print(f'Data loaded with a shape of {X.shape}')

    if args.dim:
        X = reduce_dim(X, args.dim)
        print(f'Dim of descs reduced to {X.shape}')

    embeddings = bhtsne.run_bh_tsne(X, initial_dims=X.shape[1])
    print(f'Embeddings calculated with a shape of {embeddings.shape}')

    big_picture = np.zeros((img_size * 2, img_size * 2, 3))
    center = np.array((img_size - tiny_size / 2, img_size - tiny_size / 2))
    units = (img_size - tiny_size / 2) / \
        np.max(np.absolute(embeddings), axis=0)
    for i, em in enumerate(embeddings):
        img = imgs[i]
        img = img.astype(float)
        h, w = img.shape[:2]
        pos = em * units + center
        pos = pos.astype(np.uint).tolist()
        img = cv.addWeighted(
            big_picture[pos[0]:pos[0]+h, pos[1]:pos[1]+w, :], 0.5, img, 0.5, 0)
        big_picture[pos[0]:pos[0]+h, pos[1]:pos[1]+w, :] = img
    cv.imwrite('tsne.jpg', big_picture)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='t-SNE visualizer')
    parser.add_argument('data', help='data folder')
    parser.add_argument('--descs',
                        help='path for pickle file to read descs')
    parser.add_argument('--dim', type=int,
                        help='dim for reducing dim of descs')
    parser.add_argument('--dest', default='tsne.jpg',
                        help='destination file to save tsne visualization')
    args = parser.parse_args()

    main(args)
