# -*- coding: utf-8 -*-
"""

Credit
------
Modified from Erik Linder-Norén's MIT licensed kerasGAN library,
https://github.com/eriklindernoren/Keras-GAN


"""


import scipy
import numpy as np
import os
from fnmatch import fnmatch
import cv2 as cv
import keras

from util import resize_preserving_aspect_ratio, gaussian2d
from augmentations import get_augmentation_fcn2


def points2mask(points, image_shape, use_gaussian=False, fill=False):
    """Returns n-channel mask given n x 2 array of xy-points."""

    if fill:
        assert not use_gaussian
        return cv.drawContours(image=np.zeros(image_shape[:2]),
                               contours=[cv.convexHull(np.array(points))],
                               contourIdx=-1,
                               color=1,
                               thickness=-1)

    mask = np.zeros(image_shape[:2] + (len(points),), dtype=bool)
    if use_gaussian:
        for k, (j, i) in enumerate(points):
            mask[:, :, k] = mask[:, :, k] + \
                            gaussian2d(image_shape, (j, i), variance=1)
    else:
        for k, (j, i) in enumerate(points):
            mask[i, j, k] = 1
    return mask


def pmask2points(mask):
    pts = [np.unravel_index(mask[:, :, k].argmax(), mask.shape[:2])
           for k in range(mask.shape[-1])]
    return np.array(pts)[:, ::-1]


def mask2points(mask):
    pts = []
    for k in range(mask.shape[-1]):
        pts += list(zip(*np.where(mask[:, :, k] != 0)))
    return np.array(pts)[:, ::-1]


def mark_image(image, points):
    assert 0 <= np.max(np.array(points)[:, 0]) < image.shape[1]
    assert 0 <= np.max(np.array(points)[:, 1]) < image.shape[0]
    # from copy import copy
    # img0 = copy(image)
    for k, pt in enumerate(points):

        image = cv.putText(img=image,
                           text=str(k + 1),
                           org=tuple(pt),
                           fontFace=cv.FONT_HERSHEY_SIMPLEX,
                           fontScale=.5,
                           color=(0, 0, 255),
                           thickness=2,
                           lineType=2)
    return image


def scary_shuffle_in_unison(a, b):
    # https://stackoverflow.com/questions/4601373
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def find(pattern, path):
    # https://stackoverflow.com/questions/1724693
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def read_csv(csv_path, has_headers=True):
    sample_points = dict()
    headers = None
    with open(csv_path) as f:
        for k, line in enumerate(f):

            if k == 0 and has_headers:
                headers = line.strip().split(',')
                continue

            row = line.strip().split(',')
            name = row[0]
            points = [(int(row[i]), int(row[i+1]))
                      for i in range(1, len(row)-1, 2)]
            sample_points[name] = points
    return sample_points, headers


def show_generated_pairs(generator, fx=.5, fy=.5, use_system_viewer=True):
    for image_batch, mask_batch in generator:
        for image, mask in zip(image_batch, mask_batch):
            print(image.shape)
            print(mask.shape)
            bgr = (np.flip(image, axis=2)*255).astype('uint8')
            if generator.fill:
                contours, _ = cv.findContours(image=mask.astype('uint8'),
                                              mode=cv.RETR_TREE,
                                              method=cv.CHAIN_APPROX_SIMPLE)
                visual = bgr.copy()
                cv.drawContours(visual, contours, -1, (0, 255, 0))
            else:
                assert mask.sum() == mask.shape[-1]
                pts = pmask2points(mask)
                visual = mark_image(bgr, pts)
            visual = cv.resize(visual, None, fx=fx, fy=fy)
            # visual = (visual*255).astype('uint8')
            print(visual.shape)

            if use_system_viewer:
                debug_out = '/tmp/debug_visual.jpg'
                cv.imwrite(debug_out, visual)
                os.system('open ' + debug_out)
                res = input()
                if res == 'q':
                    return
            else:
                cv.imshow('', visual)
                print(image.shape, mask.shape)

                cv.waitKey(10)
                keypress = input()
                if keypress == 'q':
                    cv.destroyAllWindows()
                    return


class PointsDataGenerator(keras.utils.Sequence):
    def __init__(self, images_dir, points_csv, subset, img_res=(128, 128),
                 image_channels=3, augment=False,
                 batch_size=2, shuffle=True, fill=False,
                 raster_then_augment=False):
        self.image_dir = images_dir
        self.points_path = points_csv
        self.img_res = img_res
        self.images = None
        self.masks = None
        self.points = None
        self.image_shape = img_res[::-1] + (image_channels,)
        self.mask_shape = None  # will be set automatically
        self.augmented = augment
        self.batch_size = batch_size
        self.subset = subset
        # self.dim = img_res
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation_fcn = get_augmentation_fcn2('truss_points')
        self.fill = fill
        self.raster_then_augment = raster_then_augment
        print('WARNING: using "scary" shuffling.  You may want to check '
              'this still works, i.e. whether masks match images.')
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""

        bs = self.batch_size
        image_batch = self.images[index*bs: (index+1)*bs]
        # mask_batch  =  self.masks[index*bs: (index+1)*bs]
        points_batch = self.points[index * bs: (index + 1) * bs]

        if self.augmented:
            augmented_mask_batch = np.empty((bs,) + self.mask_shape, dtype='float32')
            augmented_image_batch = np.empty(image_batch.shape, dtype='float32')

            for k, image, pts in zip(range(bs), image_batch, points_batch):
                keypoints = [(x, y, 0, 0) for x, y in pts]
                if self.raster_then_augment:
                    mask = points2mask(pts, image.shape, fill=self.fill)
                    augmented_data = self.augmentation_fcn(image=image,
                                                           mask=mask)
                    augmented_image = augmented_data['image']
                    augmented_mask = augmented_data['mask']
                else:
                    augmented_data = self.augmentation_fcn(image=image,
                                                           keypoints=keypoints)
                    augmented_image = augmented_data['image']
                    pts = [(x, y) for (x, y, a, s) in augmented_data['keypoints']]
                    augmented_mask = \
                        points2mask(pts, augmented_image.shape, fill=self.fill)
                augmented_mask_batch[k] = augmented_mask
                augmented_image_batch[k] = augmented_image
            image_batch = augmented_image_batch
            mask_batch = np.array(augmented_mask_batch, dtype='float32')
        else:
            mask_batch = np.empty((bs,) + self.mask_shape, dtype='float32')
            for k, pts in enumerate(points_batch):
                mask_batch[k] = \
                    points2mask(pts, self.image_shape, fill=self.fill)

        return image_batch.astype('float32') / 255, mask_batch

    def preload(self):

        unformatted_points, headers = read_csv(self.points_path)
        images = []
        # masks = []
        points = []
        subset_dir = os.path.join(self.image_dir, self.subset)
        for fn in os.listdir(subset_dir):
            fn_full = os.path.join(subset_dir, fn)
            if fn.lower().endswith('jpg') or fn.lower().endswith('.jpeg'):

                try:
                    pts = unformatted_points[fn]
                except KeyError:
                    print('Data not found for "%s" in "%s"; skipping '
                          'this sample.' % (fn, self.points_path))
                    continue

                image = self.imread(fn_full)

                # resize image and transform points along with it
                image, transform = resize_preserving_aspect_ratio(
                    image, dsize=self.img_res, border_type='constant',
                    return_coordinate_transform=True)
                pts = [np.dot(transform, [[x], [y], [1]]) for x, y in pts]
                pts = [(int(np.round(x)), int(np.round(y))) for x, y in pts]

                images.append(image)
                # masks.append(points2mask(pts, image.shape))
                points.append(pts)

        if self.fill:
            self.mask_shape = self.img_res[::-1]
        else:
            self.mask_shape = self.img_res[::-1] + (len(points[0]),)

        print('preloaded %s %s images and %s %s points'
              '' % (len(images), self.subset, len(points), self.subset))
        self.images, self.points = np.array(images), np.array(points)
        # self.masks = np.array(masks)

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB')

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        # fetch all data in subset
        if self.images is None:
            self.preload()

        # shuffle (once each epoch)
        if self.shuffle:
            rng_state = np.random.get_state()
            np.random.shuffle(self.images)
            np.random.set_state(rng_state)
            # np.random.shuffle(self.masks)
            np.random.shuffle(self.points)


if __name__ == '__main__':

    # for testing that the data is being input as expected
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('images_dir', help="Directory of images.")
    args.add_argument('points_csv', help="Path to csv w/ feature locations.")
    args.add_argument('-s', '--image_dimensions', nargs=2, type=int,
                      default=(1024, 512),
                      help="Path to csv w/ feature locations.")
    args.add_argument('--subset', default='train',
                      help="Which data set to show 'train', 'val',  or 'test'.")
    args.add_argument('-a', '--augment', default=False, action='store_true',
                      help="Use augmentation.")
    args.add_argument('-f', '--fill', default=False, action='store_true',
                      help="Masks are to be (2-D) filled convex hulls, as "
                           "opposed to 3-tensors with one channel per point.")
    args.add_argument('-r', '--raster_then_augment',
                      default=False, action='store_true',
                      help="Rasterize points to mask before, as opposed "
                           "to after, augmentation step.  This is meant for"
                           "cases where `--fill` is invoked and augmentations "
                           "do not commute with convex hull operation.")
    args = args.parse_args()

    dataset = PointsDataGenerator(images_dir=args.images_dir,
                                  points_csv=args.points_csv,
                                  img_res=args.image_dimensions,
                                  image_channels=3,
                                  subset=args.subset,
                                  batch_size=2,
                                  augment=True,
                                  fill=args.fill,
                                  raster_then_augment=args.raster_then_augment)

    show_generated_pairs(dataset, fx=1, fy=1)
