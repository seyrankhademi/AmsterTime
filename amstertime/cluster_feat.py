import time
import os
import argparse
import pickle
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def cluster(X, n_clusters=1024):
    print(f"Clustering {len(X)} descriptors" f" into {n_clusters} clusters.")
    # model = KMeans(n_clusters=n_clusters)
    model = MiniBatchKMeans(
        init="k-means++",
        n_clusters=n_clusters,
        batch_size=1000,
        n_init=10,
        max_no_improvement=10,
        verbose=0,
    )
    t0 = time.time()
    model.fit(X)
    t_elapsed = time.time() - t0
    print(f"Finished in {t_elapsed} seconds.")
    return model


def main(args):
    if not os.path.exists(args.file):
        print("File containing features couldn't found")
        exit()

    with open(args.file, "rb") as f:
        descriptors = pickle.load(f)

    X = []
    for cname in descriptors:
        for fname in descriptors[cname]:
            descs = descriptors[cname][fname]
            X.append(descs)
    X = np.concatenate(X, axis=0)
    # sample data for testing
    # indices = np.random.randint(len(X), size=100000)
    # X = X[indices]

    model = cluster(X, n_clusters=args.n_clusters)

    with open(args.model, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature clusteror")
    parser.add_argument(
        "--file", default="sift_descs.p", help="path to pickle file to read descriptors"
    )
    parser.add_argument(
        "--model",
        default="sift_kmeans.p",
        help="path to pickle file to save kmeans model",
    )
    parser.add_argument("--n-clusters", type=int, default=4096, help="# of clusters")
    args = parser.parse_args()

    main(args)
