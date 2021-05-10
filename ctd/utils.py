import os
from glob import glob
from sklearn.decomposition import PCA

classes = ['new', 'old']
image_file_exts = ('png', 'jpg')


def get_image_paths(path, num_samples=None):
    paths = []
    for ext in image_file_exts:
        paths += glob(path + '/*.' + ext)
    paths = sorted(paths)
    return paths if num_samples is None else paths[:num_samples]


def get_relative_image_paths(path, num_samples=None):
    paths = get_image_paths(path, num_samples)
    relative_paths = []
    for p in paths:
        cname, _, fname = get_path_info(p)
        relative_paths.append(os.path.join(cname, fname))
    return relative_paths


def get_path_info(path):
    splits = path.split('/')
    cname = splits[-2]
    label = splits[-1].split('.')[0]
    fname = splits[-1]
    return cname, label, fname


def get_conjugate_label(label):
    other = 'new' if label[0] == 'old' else 'old'
    return other, label[1]


def reduce_dim(X, dim):
    pca = PCA(n_components=dim)
    return pca.fit_transform(X)
