import os
import argparse
import pickle
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, models

from utils import get_path_info

avail_models = ['vgg16', 'resnet50', 'resnet101']


def main(args):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = ImageFolder(args.data, transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    selected_model = models.__dict__[args.model]
    model = selected_model(pretrained=True, progress=False)
    if 'vgg' in args.model:
        model = nn.Sequential(model.features, model.avgpool, nn.Flatten())
    else:
        modules = list(model.children())[:-1]
        model = nn.Sequential(*modules)
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)

    descriptors = {}
    pickle_path = os.path.join(args.file)
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            descriptors = pickle.load(f)
    descriptors = {cname: descriptors.get(
        cname, {}) for cname in dataset.classes}

    with torch.no_grad():
        for i, (x, _) in tqdm(enumerate(data_loader), total=len(data_loader)):
            x = x.to(device)
            scores = model(x).squeeze()
            path, _ = dataset.imgs[i]
            cname, _, fname = get_path_info(path)
            descriptors[cname][fname] = scores.to('cpu').numpy()

    with open(pickle_path, 'wb') as f:
        pickle.dump(descriptors, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN feature extractor')
    parser.add_argument('data', help='path to data set')
    parser.add_argument('--model', default=avail_models[0],
                        choices=avail_models, help='model architecture')
    parser.add_argument('--file', default='cnn.p',
                        help='path for pickle file to save descriptors')
    args = parser.parse_args()

    main(args)
