import os
import sys
import argparse
import pickle
import heapq
import numpy as np
import cv2 as cv
from scipy.spatial.distance import cdist

from .utils import reduce_dim

# To create tsne plot, download and compile https://github.com/lvdmaaten/bhtsne
# in the parent directory.
sys.path.append("../bhtsne/")
try:
    import bhtsne
except Exception as e:
    print(e)
    print("Couldn't import bhtsne. it must be in your path.")
    exit()

classes = ["new", "old"]
class_colors = {"new": (255, 0, 0), "old": (0, 255, 0)}
methods = ["exact", "nn"]


def load_img(img_path, size, center_crop=True):
    img = cv.imread(img_path)
    height, width = img.shape[:2]
    if center_crop:
        short_edge = min(height, width)
        ratio = size / short_edge
        height = int(ratio * height)
        width = int(ratio * width)
        img = cv.resize(img, (width, height), interpolation=cv.INTER_CUBIC)
        if img.shape[0] > size:
            start = int((img.shape[0] - size) / 2)
            return img[start : start + size]
        else:
            start = int((img.shape[1] - size) / 2)
            return img[:, start : start + size]
    else:
        long_edge = max(height, width)
        ratio = size / long_edge
        height = int(ratio * height)
        width = int(ratio * width)
        return cv.resize(img, (width, height), interpolation=cv.INTER_CUBIC)


def method_exact_loc(big_picture, img_size, tiny_size, embeddings, imgs):
    center = np.array((img_size - tiny_size / 2, img_size - tiny_size / 2))
    units = (img_size - tiny_size / 2) / np.max(np.absolute(embeddings), axis=0)
    for i, em in enumerate(embeddings):
        img = imgs[i]
        img = img.astype(float)
        h, w = img.shape[:2]
        pos = em * units + center
        pos = pos.astype(np.uint).tolist()
        # img = cv.addWeighted(
        #     big_picture[pos[0]:pos[0]+h, pos[1]:pos[1]+w, :], 0.25, img, 0.75, 0)
        big_picture[pos[0] : pos[0] + h, pos[1] : pos[1] + w, :] = img
    return big_picture


def method_nn(big_picture, img_size, tiny_size, embeddings, imgs):
    dist = cdist(embeddings, embeddings)
    dist -= dist.min()
    dist /= dist.max()
    nn = []
    num_imgs = len(imgs)
    for i in range(num_imgs):
        # neighbors = [j for j in range(num_imgs) if dist[i][j] < 0.03 and dist[i][j] > 1e-3]
        neighbors = [j for j in range(num_imgs) if dist[i][j] > 1e-3]
        nn.append(sorted(neighbors, key=lambda j: dist[i][j])[:5])
    idx_sorted = sorted(
        range(num_imgs), key=lambda i: embeddings[i, 0] ** 2 + embeddings[i, 1] ** 2
    )
    empty = [True] * num_imgs
    mat = np.ones((101, 101), dtype=int) * -1
    mx, my = 101 // 2, 101 // 2
    loc_x, loc_y = mx, my
    curr = idx_sorted[0]
    h = []
    heapq.heappush(h, ((loc_x - mx) ** 2 + (loc_y - my) ** 2, (curr, loc_x, loc_y)))
    mat[loc_y][loc_x] = curr
    empty[curr] = False
    while len(h):
        _, (curr, loc_x, loc_y) = heapq.heappop(h)
        next_x, next_y = min(100, loc_x + 1), loc_y
        tile(next_x, next_y, nn[curr], mat, empty, mx, my, h)
        next_x, next_y = loc_x, min(100, loc_y + 1)
        tile(next_x, next_y, nn[curr], mat, empty, mx, my, h)
        next_x, next_y = max(0, loc_x - 1), loc_y
        tile(next_x, next_y, nn[curr], mat, empty, mx, my, h)
        next_x, next_y = loc_x, max(0, loc_y - 1)
        tile(next_x, next_y, nn[curr], mat, empty, mx, my, h)

    mask = mat > -1
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    mat = mat[y0:y1, x0:x1]

    big_picture = np.ones((tiny_size * mat.shape[0], tiny_size * mat.shape[1], 3)) * 255
    for row in range(mat.shape[0]):
        for col in range(mat.shape[1]):
            idx = mat[row, col]
            if idx != -1:
                img = imgs[idx]
                h, w = img.shape[:2]
                edge_y = int((row + 0.5) * tiny_size - h / 2)
                edge_x = int((col + 0.5) * tiny_size - w / 2)
                big_picture[edge_y : edge_y + h, edge_x : edge_x + w, :] = img

    return big_picture


def tile(x, y, neighbors, mat, empty, mx, my, h):
    if mat[y][x] == -1:
        neighbor = next_nn(neighbors, empty)
        if neighbor != -1:
            empty[neighbor] = False
            mat[y][x] = neighbor
            heapq.heappush(h, ((x - mx) ** 2 + (y - my) ** 2, (neighbor, x, y)))


def next_nn(neighbors, empty):
    for n in neighbors:
        if empty[n]:
            return n
    return -1


def main(args):
    descs_path = args.descs
    if not os.path.exists(descs_path):
        print("File containing descs couldn't found")
        exit()

    with open(descs_path, "rb") as f:
        descriptors = pickle.load(f)

    tiny_size = args.tiny_size
    img_size = args.img_size

    X = []
    imgs = []
    for cname in descriptors:
        for fname in descriptors[cname]:
            img_path = os.path.join(args.data, cname, fname)
            if not os.path.exists(img_path):
                continue
            descs = descriptors[cname][fname]
            X.append(descs)
            img = load_img(img_path, tiny_size)
            cv.putText(
                img,
                fname,
                (0, 14),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                class_colors[cname],
                1,
            )
            imgs.append(img)
    X = np.array(X)
    print(f"Data loaded with a shape of {X.shape}")

    if args.dim:
        X = reduce_dim(X, args.dim)
        print(f"Dim of descs reduced to {X.shape}")

    embeddings = bhtsne.run_bh_tsne(X, initial_dims=X.shape[1])
    print(f"Embeddings calculated with a shape of {embeddings.shape}")

    big_picture = np.ones((img_size * 2, img_size * 2, 3)) * 255
    if args.method == "exact":
        big_picture = method_exact_loc(
            big_picture, img_size, tiny_size, embeddings, imgs
        )
    elif args.method == "nn":
        big_picture = method_nn(big_picture, img_size, tiny_size, embeddings, imgs)
    cv.imwrite("tsne.png", big_picture)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="t-SNE visualizer")
    parser.add_argument("data", help="data folder")
    parser.add_argument("--descs", help="path for pickle file to read descs")
    parser.add_argument("--dim", type=int, help="dim for reducing dim of descs")
    parser.add_argument(
        "--dest", default="tsne.jpg", help="destination file to save tsne visualization"
    )
    parser.add_argument("--tiny-size", type=int, default=64, help="size of each image")
    parser.add_argument("--img-size", type=int, default=1024, help="output image size")
    parser.add_argument(
        "--method", choices=methods, default=methods[0], help="image tiling method"
    )
    args = parser.parse_args()

    main(args)
