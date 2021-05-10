import numpy as np
import pandas as pd
import os
import argparse
from utils import get_relative_image_paths

classes = ['new', 'old']
tasks = ['verification', 'retrieval']


def generate_verification_task(root):
    new_images = get_relative_image_paths(os.path.join(root, 'new'))
    old_images = get_relative_image_paths(os.path.join(root, 'old'))
    num_images = len(new_images)
    samples = []
    # add positive samples
    for i in range(num_images):
        image1 = new_images[i]
        image2 = old_images[i]
        samples.append((image1, image2, 1))
    # add negative samples
    rand_samples = np.random.permutation(num_images)
    for i in range(num_images):
        rand_sample = rand_samples[i]
        # change sample if they are same samples
        while i == rand_sample:
            rand_sample = np.random.randint(num_images)
        image1 = new_images[i]
        image2 = old_images[rand_sample]
        samples.append((image1, image2, 0))
    # permute samples to scatter positive and negative samples
    # samples = np.random.permutation(samples)

    return samples, ['image1', 'image2', 'y']


def generate_retrieval_task(root):
    new_images = get_relative_image_paths(os.path.join(root, 'new'))
    old_images = get_relative_image_paths(os.path.join(root, 'old'))
    samples = []
    # add new images as query with target class of old to search into it
    samples.extend([(q, 'old') for q in new_images])
    # now, add old images as query with target class of new to search into it
    samples.extend([(q, 'new') for q in old_images])

    return samples, ['query', 'target']


def main(args):
    if args.task == 'verification':
        data, columns = generate_verification_task(args.data)
    elif args.task == 'retrieval':
        data, columns = generate_retrieval_task(args.data)

    # save samples
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(args.task_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task generator')
    parser.add_argument('data', help='path to data set')
    parser.add_argument('--task', default=tasks[0], choices=tasks,
                        help='name of task to generate')
    parser.add_argument('--task-file', help='file to save the task')
    args = parser.parse_args()

    main(args)
