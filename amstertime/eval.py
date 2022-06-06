import os
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

from .utils import get_path_info


tasks = ["verification", "retrieval"]
dist_types = ["l2", "cosine"]


def calculate_metrics(results):
    _, k = results.shape
    P = np.cumsum(results, axis=1) / np.cumsum(np.ones_like(results), axis=1)
    AP = np.sum(P * results, axis=1)
    mAP = np.mean(AP)
    acc = np.mean(results, axis=0)
    top1 = np.sum(acc[:1])
    top5 = 0
    if k >= 5:
        top5 = np.sum(acc[:5])
    return mAP, top1, top5


def calculate_dists(samples, descriptors, dist_type="l2"):
    print("# of queries in the task file:", len(samples))
    feats1 = []
    feats2 = []
    y = []
    pairs = []
    for _, s in samples.iterrows():
        cname1, _, fname1 = get_path_info(s["image1"])
        if fname1 not in descriptors[cname1]:
            print(f"Descriptor not found for file: {s['image1']}")
            continue
        cname2, _, fname2 = get_path_info(s["image2"])
        if fname2 not in descriptors[cname2]:
            print(f"Descriptor not found for file: {s['image2']}")
            continue
        feats1.append(descriptors[cname1][fname1])
        feats2.append(descriptors[cname2][fname2])
        y.append(s["y"])
        pairs.append((s["image1"], s["image2"], s["y"]))
    feats1 = np.stack(feats1)
    feats2 = np.stack(feats2)
    print("# of queries:", feats1.shape[0])
    print("# of features:", feats1.shape[1])

    if dist_type == "l2":
        dists = np.sum((feats1 - feats2) ** 2, axis=1)
    elif dist_type == "cosine":
        dists = np.abs(
            1
            - np.sum(feats1 * feats2, axis=1)
            / np.sqrt(np.sum(feats1**2, axis=1) * np.sum(feats2**2, axis=1))
        )
    else:
        raise Exception(f"Unknown distance type: {dist_type}")
    y = np.array(y).astype(np.int64)
    return dists, y, pairs


def eval_verification(samples, descriptors, visualize=False, dist_type="l2"):
    dists, y, pairs = calculate_dists(samples, descriptors, dist_type)

    np.random.seed(1234)

    # normalize the distances
    X = dists.reshape(-1, 1)
    X = X - np.mean(X, axis=0)
    X = X / (np.std(X, axis=0) + 1e-10)
    # normalizer = StandardScaler()
    # X = normalizer.fit_transform(X)

    y_true = [p[2] for p in pairs]
    # because X contains the distances, there is no minus in exponential
    y_score = 1 / (1 + np.exp(X.ravel()))
    auc = roc_auc_score(y_true, y_score)

    # pick mean as threshold
    mean_dist = np.mean(X)
    predictions_mean = X < mean_dist

    if visualize:
        # scatter distances with positive and negative markers
        a = sorted(list(zip(X, y)))
        y_pos, x_pos, y_neg, x_neg = [], [], [], []
        for i, (d, yy) in enumerate(a):
            if yy == 1:
                y_pos.append(d[0])
                x_pos.append(i)
            else:
                y_neg.append(d[0])
                x_neg.append(i)

        _, ax = plt.subplots()
        plt.scatter(x_pos, y_pos, marker="o", label="Positive pairs")
        plt.scatter(x_neg, y_neg, marker="v", label="Negative pairs")
        ax.set_ylabel("Distance")
        ax.set_xlabel("Pairs")
        ax.legend()
        plt.show()

        # show how acc changes with distance threshold
        # hand pick best threshold
        accs = []
        min_d = np.min(X)
        max_d = np.max(X)
        thresholds = [min_d + i * (max_d - min_d) / 1000 for i in range(1000)]
        for threshold in thresholds:
            predictions = X < threshold
            results = (predictions == y.reshape(-1, 1)).astype(np.float)
            acc, _, _ = calculate_metrics(results)
            accs.append(acc)
        _, ax = plt.subplots()
        ax.plot(thresholds, accs)
        ax.set_ylabel("acc")
        ax.set_xlabel("Threshold")
        plt.show()

    true_positives = [
        (p, pair) for p, pair in zip(predictions_mean, pairs) if p and pair[2]
    ]
    true_negatives = [
        (p, pair)
        for p, pair in zip(predictions_mean, pairs)
        if (not p) and (not pair[2])
    ]
    false_positives = [
        (p, pair) for p, pair in zip(predictions_mean, pairs) if p and (not pair[2])
    ]
    false_negatives = [
        (p, pair) for p, pair in zip(predictions_mean, pairs) if (not p) and pair[2]
    ]

    tp = len(true_positives)
    tn = len(true_negatives)
    fp = len(false_positives)
    fn = len(false_negatives)
    t = tp + tn
    f = fp + fn
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * pre * rec / (pre + rec)
    acc = t / (t + f)

    print("# true positives : {:4d}".format(tp))
    print("# true negatives : {:4d}".format(tn))
    print("# false positives: {:4d}".format(fp))
    print("# false negatives: {:4d}".format(fn))
    print("# total pairs    : {:4d}".format(t + f))
    print()
    print("precision: {:0.4f}".format(pre))
    print("recall   : {:0.4f}".format(rec))
    print("f1       : {:0.4f}".format(f1))
    print("acc      : {:0.4f}".format(acc))
    print("roc_auc  : {:0.4f}".format(auc))

    return acc, predictions_mean, pairs


def calculate_query_results(queries, data, query_labels, data_labels, k, dist_type):
    # create a model
    model = NearestNeighbors(
        n_neighbors=k, n_jobs=-1, metric="minkowski" if dist_type == "l2" else dist_type
    )
    model.fit(data)

    # query on the model
    neighbors = model.kneighbors(queries, return_distance=False)

    # calculate results
    results = []
    for i, n in enumerate(neighbors):
        result = np.array([query_labels[i] == data_labels[j] for j in n])
        results.append(result)

    return results


def eval_retrieval(samples, descriptors, k, dist_type):
    X = {"new": [], "old": []}
    y = {"new": [], "old": []}
    for _, s in samples.iterrows():
        cname, label, fname = get_path_info(s["query"])
        if fname not in descriptors[cname]:
            print(f"Descriptor not found for file: {s['query']}")
            continue
        X[cname].append(descriptors[cname][fname])
        y[cname].append(label)

    X_new = np.array(X["new"])
    X_old = np.array(X["old"])
    y_new = y["new"]
    y_old = y["old"]

    # query new images on old images
    old_results = calculate_query_results(X_old, X_new, y_old, y_new, k, dist_type)
    old_results = np.stack(old_results).astype(float)
    old_metrics = calculate_metrics(old_results)
    print("Old")
    print("mAP", old_metrics[0])
    print("Top1", old_metrics[1])
    print("Top5", old_metrics[2])
    # do the same but this time old image on new ones
    new_results = calculate_query_results(X_new, X_old, y_new, y_old, k, dist_type)
    new_results = np.stack(new_results).astype(float)
    new_metrics = calculate_metrics(new_results)
    print("New")
    print("mAP", new_metrics[0])
    print("Top1", new_metrics[1])
    print("Top5", new_metrics[2])

    results = np.concatenate((new_results, old_results), axis=0)

    # mAPs = []
    # for i in range(1, results.shape[1]):
    #     mAP, _, _ = calculate_metrics(results[:, :i])
    #     mAPs.append(mAP)
    # plt.plot(mAPs)
    # plt.show()
    mAP, top1, top5 = calculate_metrics(results)
    print("All")
    print("mAP", mAP)
    print("Top1", top1)
    print("Top5", top5)
    return mAP


def main(args):
    descriptors_path = args.feats
    if not os.path.exists(descriptors_path):
        print("File containing descriptors couldn't found")
        exit()
    with open(descriptors_path, "rb") as f:
        descriptors = pickle.load(f)

    task_path = args.file
    if not os.path.exists(task_path):
        print("Task file couldn't found")
        exit()
    samples = pd.read_csv(task_path)

    if args.task == "verification":
        eval_verification(
            samples, descriptors, visualize=False, dist_type=args.dist_type
        )
    elif args.task == "retrieval":
        eval_retrieval(samples, descriptors, args.k, dist_type=args.dist_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluator")
    parser.add_argument(
        "--task",
        default=tasks[0],
        choices=tasks,
        help="name of task to run evaluation for",
    )
    parser.add_argument("--file", required=True, help="path to task file")
    parser.add_argument(
        "--feats", required=True, help="path to pickle file to read features"
    )
    parser.add_argument("--k", type=int, default=1, help="k in p@k")
    parser.add_argument(
        "--dist_type", default=dist_types[0], choices=dist_types, help="distance type"
    )
    args = parser.parse_args()

    main(args)
