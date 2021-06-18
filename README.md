# Cross-Time Dataset
This repository contains a dataset of image pairs from historical Amsterdam and corresponding current street-view of the same place with evaluation code. The pairs are collected using crowd-sourcing methods. For a given historical image, which contains a view of a place in Amsterdam and has a provided approximate location on the map, volunteers find the best shot point of that image by moving around on the street-view application and once they believe that they find the best point a screenshot is taken. After this process, each found pair passes from moderator verification. There are currectly 1231 verified pairs. The following is a sample pair in the dataset:

| ![Historical image from Amsterdam Beeldbank](http://images.memorix.nl/ams/thumb/widget640/816cac37-15a1-4740-607a-6795df87e0f5.jpg) | ![Corresponding street-view from Mapillary](.github/sample_mapillary.png) |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| *Historical image from [Amsterdam Beeldbank](https://archief.amsterdam/beeldbank)* |          *Corresponding street-view from Mapillary*          |

And the visualization of the dataset using t-SNE:

![The visualization of the dataset with t-SNE](.github/tsne.jpg)

## Requirements

The code works on `python >= 3.8`.

To install requirements:

```
pip install -r requirements.txt
```

## Tasks

### Verification

The aim of this task is that for a given image pair of historical image and street-view image determining whether the pair is positive which means that they are taken from same place.

### Retrieval

This task is to find corresponding image in the mapillary images for a given historical image or vice versa.

## Usage

### Evaluation

Evaluation code works with pickle file of features and calculates metrics for either verification or retrieval task. The pickle file has to be in the following structure:

```
{
  'new': {
    'image_1.png': feature_array_for_image1,
    'image_2.png': feature_array_for_image2,
    ...
  },
  'old': {
    'image_1.png': feature_array_for_image1,
    'image_2.png': feature_array_for_image2,
    ...
  }
}
```

Usage of evaluation code:

```bash
python -m ctd.eval --help
```

```
usage: eval.py [-h] [--task {verification,retrieval}] [--file FILE] [--feats FEATS] [--k K] [--dist_type {l2,cosine}]

Evaluator

optional arguments:
  -h, --help            show this help message and exit
  --task {verification,retrieval}
                        name of task to run evaluation for
  --file FILE           path for task file
  --feats FEATS         path for pickle file to read features
  --k K                 k in p@k
  --dist_type {l2,cosine}
                        distance type
```

Example usage to calculate metrics for verification task for BoW features computed using SIFT descriptors:

```
python ctd.eval --task verification --file tasks/verification.csv --feats features/sift_128.p 
```

## Results

Many methods and pre-trained models are used to extract features and the features are used for verification and retrieval tasks. We shared all of the features extracted in [this link](https://data.4tu.nl/account/articles/14572644). The following table shows the details of files in the link:

| File                                                         | Method / Model                                               | Paper                                                        | Codebase                                            |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------------------------------------- |
| [sift_128.p](https://data.4tu.nl/ndownloader/files/27962772) | SIFT                                                         | https://link.springer.com/article/10.1023/B:VISI.0000029664.99615.94 | https://github.com/seyrankhademi/cross-time-dataset |
| [lift_piccadilly_128.p](https://data.4tu.nl/ndownloader/files/28329594) | LIFT                                                         | https://link.springer.com/chapter/10.1007%2F978-3-319-46466-4_28 | https://github.com/cvlab-epfl/tf-lift               |
| [vgg16_imagenet.p](https://data.4tu.nl/ndownloader/files/27962799) | VGG-16                                                       | https://arxiv.org/abs/1409.1556v4                            | https://github.com/pytorch/vision                   |
| [resnet50_imagenet.p](https://data.4tu.nl/ndownloader/files/27962766) | ResNet-50                                                    | https://arxiv.org/abs/1512.03385                             | https://github.com/pytorch/vision                   |
| [resnet101_imagenet.p](https://data.4tu.nl/ndownloader/files/27962769) | ResNet-101                                                   | https://arxiv.org/abs/1512.03385                             | https://github.com/pytorch/vision                   |
| [netvlad_pittsburgh250k.p](https://data.4tu.nl/ndownloader/files/28329639) | [NetVLAD w/ ResNet-18](https://drive.google.com/open?id=17luTjZFCX639guSVy00OUtzfTQo4AMF2) | https://arxiv.org/abs/1511.07247                             | https://github.com/Nanne/pytorch-NetVlad            |
| [simsiam_resnet18_scratch.p](https://data.4tu.nl/ndownloader/files/27962790) | SimSiam w/ ResNet-18                                         | https://arxiv.org/abs/2011.10566                             | Self-reproduction (Code will be available)          |
| [simsiam_resnet18_imagenet.p](https://data.4tu.nl/ndownloader/files/27962784) | SimSiam w/ ResNet-18                                         | https://arxiv.org/abs/2011.10566                             | Self-reproduction (Code will be available)          |
| [simsiam_netvlad_pittsburgh250k.p](https://data.4tu.nl/ndownloader/files/28329642) | SimSiam w/ NetVLAD                                           | https://arxiv.org/abs/2011.10566                             | Self-reproduction (Code will be available)          |
| [ap_gem.p](https://data.4tu.nl/ndownloader/files/27962757)   | [Resnet101-AP-GeM](https://drive.google.com/open?id=1UWJGDuHtzaQdFhSMojoYVQjmCXhIwVvy) | https://arxiv.org/abs/1906.07589                             | https://github.com/naver/deep-image-retrieval       |
| [triplet_resnet18_imagenet.p](https://data.4tu.nl/ndownloader/files/27962793) | Triplet ResNet-18                                            | http://www.bmva.org/bmvc/2016/papers/paper119/index.html     | Self-reproduction (Code will be available)          |

### Results for verification task

The following table shows the results for verification task using the features given in the above table:

|           | SIFT | LIFT | VGG-16 | ResNet-50 | ResNet-101 | NetVLAD w/ ResNet-18 | SimSiam w/ ResNet-18 | SimSiam w/ ResNet-18 | SimSiam w/ NetVLAD | Resnet101-AP-GeM | Triplet ResNet-18 |
| --------- | ---- | ---- | ------ | --------- | ---------- | -------------------- | -------------------- | -------------------- | ------------------ | ---------------- | ----------------- |
| Precision | 0.55 | 0.51 | 0.58   | 0.63      | 0.63       | 0.83                 | 0.54                 | 0.52                 | 0.70               | 0.88             | 0.81              |
| Recall    | 0.73 | 0.74 | 0.62   | 0.66      | 0.67       | 0.80                 | 0.80                 | 0.71                 | 0.74               | 0.78             | 0.91              |
| F1        | 0.62 | 0.61 | 0.60   | 0.65      | 0.65       | 0.82                 | 0.65                 | 0.60                 | 0.72               | 0.83             | 0.86              |
| Accuracy  | 0.56 | 0.52 | 0.58   | 0.63      | 0.64       | 0.82                 | 0.56                 | 0.52                 | 0.71               | 0.84             | 0.85              |

### Results for retrieval task

The following table shows the results for retrieval task using the features given in the above table:

| Query Image | Metric  | SIFT | LIFT | VGG-16 | ResNet-50 | ResNet-101 | NetVLAD w/ ResNet-18 | SimSiam w/ ResNet-18 | SimSiam w/ ResNet-18 | SimSiam w/ NetVLAD | Resnet101-AP-GeM | Triplet ResNet-18 |
| ----------- | ------- | ---- | ---- | ------ | --------- | ---------- | -------------------- | -------------------- | -------------------- | ------------------ | ---------------- | ----------------- |
| Historical  | mAP@128 | 0.03 | 0.02 | 0.08   | 0.06      | 0.05       | 0.26                 | 0.03                 | 0.07                 | 0.29               | 0.35             | 0.40              |
|             | Top1    | 0.01 | 0.01 | 0.06   | 0.03      | 0.03       | 0.17                 | 0.01                 | 0.04                 | 0.21               | 0.24             | 0.28              |
|             | Top5    | 0.03 | 0.04 | 0.11   | 0.07      | 0.07       | 0.33                 | 0.03                 | 0.07                 | 0.36               | 0.48             | 0.51              |
| Street-view | mAP@128 | 0.02 | 0.02 | 0.06   | 0.05      | 0.06       | 0.26                 | 0.02                 | 0.07                 | 0.29               | 0.34             | 0.39              |
|             | Top1    | 0.01 | 0.01 | 0.04   | 0.03      | 0.03       | 0.18                 | 0.00                 | 0.03                 | 0.20               | 0.24             | 0.28              |
|             | Top5    | 0.02 | 0.03 | 0.08   | 0.06      | 0.07       | 0.34                 | 0.03                 | 0.09                 | 0.38               | 0.44             | 0.51              |
| All         | mAP@128 | 0.02 | 0.02 | 0.07   | 0.05      | 0.05       | 0.26                 | 0.02                 | 0.07                 | 0.29               | 0.35             | 0.40              |
|             | Top1    | 0.01 | 0.01 | 0.05   | 0.03      | 0.03       | 0.18                 | 0.01                 | 0.04                 | 0.21               | 0.24             | 0.28              |
|             | Top5    | 0.03 | 0.03 | 0.09   | 0.07      | 0.07       | 0.33                 | 0.03                 | 0.08                 | 0.37               | 0.46             | 0.51              |

