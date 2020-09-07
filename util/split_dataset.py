""" Split a dataset into a train/val/test subsets.

Usage
-----
* Typical usage::

    $ python split_dataset.py unsplit_dir split_dir 5 5


* To remove any files encountered that can't be read with `cv.imread()`,
use the `-u` or `--remove_unreadable` flag::

    $ python split_dataset.py -u unsplit_dir split_dir 5 5


"""


import os
import cv2 as cv
import numpy as np
from shutil import copytree


def is_image(fn, extensions=('jpg', 'jpeg', 'png')):
    return os.path.splitext(fn)[1][1:].lower() in extensions


def move_random(src_dir, dst_dir, n=1, extensions=None):

    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)

    files = os.listdir(src_dir)
    if extensions:
        files = [f for f in files if is_image(f, extensions)]

    if 0 < n < 1:
        n = int(len(files) * n)

    for fn in np.random.choice(files, n, replace=False):
        os.rename(os.path.join(src_dir, fn),
                  os.path.join(dst_dir, fn))


def move_random_images(src_dir, dst_dir, n=1,
                       extensions=('jpg', 'jpeg', 'png')):
    move_random(src_dir=src_dir, dst_dir=dst_dir, n=n,
                extensions=extensions)


def make_subset(split_dir, subset, n_subset, remove_unreadable=False,
                subdirectory_labels=True):
    train_dir = os.path.join(split_dir, 'train')
    subset_dir = os.path.join(split_dir, subset)

    # setup `subset_dir`
    os.mkdir(subset_dir)
    if subdirectory_labels:
        categories = os.listdir(train_dir)
        for category in categories:
            os.mkdir(os.path.join(subset_dir, category))
    else:
        categories = ['.']

    for category in categories:
        if not os.path.isdir(os.path.join(train_dir, category)):
            continue

        # remove unreadable images
        if remove_unreadable:
            images = [f for f in os.listdir(os.path.join(train_dir, category))]
            for fn in images:
                img = cv.imread(os.path.join(train_dir, category, fn))
                try:
                    img.shape
                except AttributeError:
                    os.remove(os.path.join(train_dir, category, fn))

        # move `n_subset` images of this subset to
        images = [f for f in os.listdir(os.path.join(train_dir, category))]
        for fn in np.random.choice(images, n_subset, replace=False):
            os.rename(os.path.join(train_dir, category, fn),
                      os.path.join(subset_dir, category, fn))


def split_dataset(data_dir, out_dir, n_val, n_test, remove_unreadable=False,
                  subdirectory_labels=True):

    # copy dataset to training dir
    os.makedirs(out_dir)
    copytree(data_dir, os.path.join(out_dir, 'train'))

    if n_val:
        make_subset(out_dir, 'val', n_val, remove_unreadable,
                    subdirectory_labels=subdirectory_labels)
    if n_test:
        make_subset(out_dir, 'test', n_test, remove_unreadable,
                    subdirectory_labels=subdirectory_labels)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        help="Root dir of original unsplit dataset.  Assumed to be "
             "divided into subdirectories by class.")
    parser.add_argument(
        "out_dir",
        help="Where to store new split dataset.")
    parser.add_argument(
        "n_val", type=int,
        help="The number of images per class to put in the validation set.")
    parser.add_argument(
        "n_test", type=int,
        help="The number of images per class to put in the validation set.")
    parser.add_argument(
        '-u', '--remove_unreadable', default=False, action='store_true',
        help="Don't include any unreadable images encountered.")
    parser.add_argument(
        '-f', '--filename_labels', default=False, action='store_true',
        help="No subdirectories.")
    args = parser.parse_args()

    split_dataset(data_dir=args.data_dir,
                  out_dir=args.out_dir,
                  n_val=args.n_val,
                  n_test=args.n_test,
                  remove_unreadable=args.remove_unreadable,
                  subdirectory_labels=not args.filename_labels)
