import os
import argparse
import pickle
import numpy as np


def compute_bovw(X, model):
    if len(X) == 0:
        hist = np.zeros((len(model.cluster_centers_), ))
    else:
        clusters = model.predict(X)
        hist, _ = np.histogram(clusters, bins=len(model.cluster_centers_))
    # normalize the histogram
    hist = hist.astype(np.float) / len(X)
    return hist


def main(args):
    if not os.path.exists(args.file):
        print('File containing features couldn\'t found')
        exit()

    if not os.path.exists(args.model):
        print('File containing kmeans model couldn\'t found')
        exit()

    with open(args.file, 'rb') as f:
        descriptors = pickle.load(f)
    with open(args.model, 'rb') as f:
        model = pickle.load(f)
        print(model)

    features = {}
    for cname in descriptors:
        f = {}
        for fname in descriptors[cname]:
            descs = descriptors[cname][fname]
            f[fname] = compute_bovw(descs, model)
        features[cname] = f

    with open(args.bovw, 'wb') as f:
        pickle.dump(features, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BOVW calculator')
    parser.add_argument('--file', default='sift_descs.p',
                        help='path to pickle file to read descriptors')
    parser.add_argument('--model', default='sift_kmeans.p',
                        help='path to pickle file to read kmeans model')
    parser.add_argument('--bovw', default='sift.p',
                        help='path to pickle file to save bovw features')
    args = parser.parse_args()

    main(args)
