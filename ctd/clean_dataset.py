import os
import argparse
from utils import classes


def main():
    nums = []
    with open(args.file, 'r') as f:
        for line in f:
            nums.append(int(line))

    for c in classes:
        for n in nums:
            img_path = os.path.join(args.data, c, '{:04d}.png'.format(n))
            if os.path.exists(img_path):
                print(img_path)
                os.remove(img_path)
            img_path = os.path.join(args.data, c, '{:04d}.jpg'.format(n))
            if os.path.exists(img_path):
                print(img_path)
                os.remove(img_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data cleaner')
    parser.add_argument('data', help='data root')
    parser.add_argument('--file',
                        help='path to file where image numbers to be deleted are listed')
    args = parser.parse_args()

    main(args)
