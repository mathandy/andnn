#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Review image segmentations.

This script makes it fast and easy to review and hand-score image
segmentations.  A text file, "reviewed.txt" will be created and appended
to with your scores as you go.  Further instructions will be provided
when the script is run.

Requirements
------------
* Numpy
* OpenCV (tested with version 3.4.1)

Recommended Usage
-----------------
Call this script from a directory containing an two subdirectories named
"images" and "segmentations" as well as (if relevant) the dictionary
of affine transforms, "affine_transforms.npz", created in the
post-processing steps (see `segmentation_postprocessing.py`).::

    $ cd path/to/parent_dir_of_images_and_segmentations
    $ python path/to/review_segmentations.py


For more options::

    $ python review_segmentations.py -h


"""


from __future__ import print_function, division
import os
from sys import platform
from glob import glob
import cv2 as cv  # tested with version 3.4.1
import numpy as np
from itertools import product as cartesian_product
from shutil import copyfile
try:
    input = raw_input
except:
    pass


def getch():
    """Gets a single character from standard input.
    I.e. it's like `input()` but you don't have to press enter.

    Credit: https://code.activestate.com/recipes/134892/"""
    try:
        import msvcrt  # for MS-Windows compatibility
        return msvcrt.getch()
    except ImportError:
        pass
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def scale_with_padding(image, dsize, pad_with=0):
    if image.ndim == 3 and isinstance(pad_with, int):
        pad_with = (pad_with, pad_with, pad_with)
    ssize = image.shape[:2][::-1]
    x_diff, y_diff = np.array(ssize) - dsize
    assert not (x_diff % 2) and not (y_diff % 2)
    new_image = cv.resize(image, dsize)

    rows = [[pad_with]*dsize[0]]*(x_diff//2)
    new_image = np.vstack((rows, new_image, rows))
    cols = np.hstack([[[pad_with]]*ssize[1]]*(y_diff//2))
    new_image = np.hstack((cols, new_image, cols))
    return new_image


def overlay(image, mask, contrast=0.2):
    if mask.ndim == 2:
        mask = np.stack((mask, mask, mask), 2)
    mask = (mask > 0) * 1

    if mask.shape[0] > image.shape[0]:
        mask = mask[:image.shape[0]]

    return (image*(1 * mask + contrast * (np.logical_not(mask)))).astype('uint8')


def get_sample_pair_by_name(name, multiple_file_warning=False,
                            return_filenames=False):
    images = glob(os.path.join('images', name) + '.*')
    segmentation_masks = glob(os.path.join('segmentations', name) + '.*')
    if multiple_file_warning and len(images) > 1:
        print("Warning: multiple images begin with %s" % name)
        for image in images:
            print(image)
    elif not images:
        raise IOError("No image found beginning with name %s" % name)
    if multiple_file_warning and len(segmentation_masks) > 1:
        print("Warning: multiple segmentation masks begin with %s" % name)
        for image in images:
            print(image)
    elif not segmentation_masks:
        raise IOError("No segmentation mask found beginning with name %s" % name)
    if return_filenames:
        return images[0], segmentation_masks[0]
    return cv.imread(images[0]), cv.imread(segmentation_masks[0])


def affine_inv(transform):
    # y = ax + b
    # x = a^(-1)*(y - b)
    A_inv, b = np.linalg.inv(transform[:, :2]), transform[:, 2]
    return np.hstack((A_inv, -np.dot(A_inv, b).reshape(2, 1)))


def visualize(name=None, visual_out='review_visual.png', transform=None,
              scale_dsize=None, contrast=0.2, polygons=None, gt_polygons=None,
              debug=False):
    if name is None:
        name = np.random.choice(os.listdir('images'))[:-4]
    img, seg = get_sample_pair_by_name(name, True)

    if debug:
        from ipdb import set_trace
        set_trace()

    if scale_dsize is not None:
        seg = scale_with_padding(seg, scale_dsize)

    if transform is not None:
        overlay_seg = cv.warpAffine(seg, affine_inv(transform),
                                    img.shape[:2][::-1])
        overlay_img = img
    else:
        overlay_img, overlay_seg = img, seg
    # seg = 1 * (seg[:img.shape[0]] > 0)
    overlay_image = overlay(overlay_img, overlay_seg, contrast=contrast)

    if seg.ndim == 2:
        seg = np.stack([seg]*3, 2)
    # seg = seg * 255

    if polygons is not None:
        for p in polygons:
            p = p.astype(int)
            overlay_image = cv.drawContours(overlay_image, [p], -1, (0, 255, 0), 1)
            seg = cv.drawContours(seg.astype('uint8'), [p], -1, (0, 255, 0), 1)
            img = cv.drawContours(img, [p], -1, (0, 255, 0), 1)
    if gt_polygons is not None:
        for p in gt_polygons:
            p = p.astype(int)
            overlay_image = cv.drawContours(overlay_image, [p], -1, (0, 0, 255), 1)
            seg = cv.drawContours(seg.astype('uint8'), [p], -1, (0, 0, 255), 1)
            img = cv.drawContours(img, [p], -1, (0, 0, 255), 1)

    def intersection_over_union(p, q):
        p_filled = cv.fillConvexPoly(np.zeros(seg.shape[:2]), p.astype(int), 1)
        q_filled = cv.fillConvexPoly(np.zeros(seg.shape[:2]), q.astype(int), 1)
        return np.logical_and(p_filled, q_filled).sum() / np.logical_or(p_filled, q_filled).sum()

    # print intersection(s) over union(s)
    if polygons is not None and gt_polygons is not None:
        ious = [intersection_over_union(p, q) for p, q in
                cartesian_product(polygons, gt_polygons)]
        ious = sorted(ious)[::-1][:max(len(polygons), len(gt_polygons))]

        print('Image Name:', name)
        print('IoUs:', ', '.join(str(iou) for iou in ious))

    if img.shape[0] < img.shape[1]:
        tiled = np.vstack((img, overlay_image, seg))
    else:
        tiled = np.hstack((img, overlay_image, seg))
    cv.imwrite(visual_out, tiled)

    if platform == "linux" or platform == "linux2":
        # os.system('xdg-open %s &' % visual_out)
        os.system('xdg-open %s &' % visual_out)
    elif platform == "darwin":
        os.system('open %s' % visual_out)
    else:
        print('file stored at %s' % visual_out)


def review(root_dir=None, visual_out='review_visual.png',
           transforms_path='affine_transforms.npz', shuffle=True,
           allow_errors=True, polygons_path=None, gt_polygons_path=None,
           skip_true_negatives=False):

    try:
        transforms = np.load(transforms_path)
    except:
        print("Problem loading transforms_path = %s" % transforms_path)
        transforms = None
        transform = None

    polygons = None
    gt_polygons = None
    image_polygons = None
    image_gt_polygons = None
    if polygons_path is not None:
        polygons = np.load(polygons_path)

    if gt_polygons_path is not None:
        gt_polygons = np.load(gt_polygons_path)

    _working_dir = os.getcwd()
    if root_dir is None:
        root_dir = _working_dir
    os.chdir(root_dir)
    images = set(os.path.splitext(f)[0] for f in os.listdir('images'))
    if os.path.isfile('reviewed.txt'):
        with open('reviewed.txt', 'r') as f:
            reviewed = [line.split(',')[0] for line in f]
        up_for_review = images - set(reviewed)
    else:
        up_for_review = images
    up_for_review = list(up_for_review)

    if shuffle:
        np.random.shuffle(up_for_review)

    commands = {'1': 'Include',
                '2': 'Maybe Include',
                '3': 'Disclude',
                's': 'Skip',
                'd': 'debug',
                'x': 'special mark to remember by',
                'w': 'save visual',
                'q': 'quit'}

    def list_commands(name=None):
        if name is not None:
            print("Image Name:", name)
        print("Commands:")
        for key, val in commands.items():
            print(key + ':', val)
        print('Note: if you press a wrong key, this list will reappear.')

    print("\n%s images up for review.\n" % len(up_for_review))
    for i, name in enumerate(up_for_review):
        if name == 'via_region_data':
            continue
        try:
            if transforms is not None:
                transform = transforms[name]
            if polygons:
                image_polygons = polygons[name]
            if gt_polygons:
                image_gt_polygons = gt_polygons[name]
            if skip_true_negatives and polygons and gt_polygons:
                if not len(image_polygons) and not len(image_gt_polygons):
                    print('skipping true negative %s.' % name)
                    continue
            visualize(name, visual_out, transform, polygons=image_polygons,
                      gt_polygons=image_gt_polygons)
        except Exception as e:
            print("ERROR:", e)
            if allow_errors:
                continue
            else:
                raise
        print('image %s/%s: ' % (i, len(up_for_review)))
        user_response = getch()
        while user_response not in commands.keys():
            list_commands(name)
            user_response = getch()
        print(name, user_response)
        if user_response == 's':
            continue
        elif user_response == 'w':
            default_save_path = os.path.join(os.getcwd(), 'visual_' + name + '.png')
            save_path = input('Enter save path (defaults to %s):' % default_save_path)
            copyfile(visual_out, save_path if save_path else default_save_path)
        elif user_response == 'q':
            quit()
        elif user_response == 'd':
            visualize(name, visual_out, debug=True)
        with open('reviewed.txt', 'a+') as reviewed_txt:
            reviewed_txt.write(name + ',' + user_response + '\n')
    os.chdir(_working_dir)


if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("-r", "--root_dir", default=os.getcwd(),
                      help="directory containing 'images' and "
                           "'segmentations' subdirectories")
    args.add_argument("-n", "--name", default=None,
                      help="visualize just this example")
    args.add_argument("-o", "--visual_out", default='review_visual.png',
                      help="where to store the most recent "
                           "visualization output")
    args.add_argument("-p", "--polygons_path",
                      default=None,
                      help="path to npz storing ground truth polygons to draw")
    args.add_argument("-g", "--gt_polygons_path",
                      default=None,
                      help="path to npz storing predicted polygons to draw")
    args.add_argument("-t", "--transforms_path",
                      default='affine_transforms.npz',
                      help="affine transforms npz file")
    args.add_argument("-e", "--allow_errors", default=True, action='store_false',
                      help='Whether to stop and show full traceback '
                           'if errors are encountered.')
    args.add_argument("-s", "--skip_true_negatives", default=False,
                      action='store_true',
                      help="Skip the true negatives.")
    args = vars(args.parse_args())

    # check for polygons in `root_dir`
    polygons_path = args['polygons_path']
    gt_polygons_path = args['gt_polygons_path']
    if polygons_path is None:
        default_path = os.path.join(args['root_dir'], 'predicted_polygons.npz')
        if os.path.exists(default_path):
            polygons_path = default_path
    if gt_polygons_path is None:
        default_path = os.path.join(args['root_dir'], 'gt_polygons.npz')
        if os.path.exists(default_path):
            gt_polygons_path = default_path
    if polygons_path and gt_polygons_path:
        print("Ground truth is red, predictions are green.")

    if args['name'] is not None:
        if polygons_path:
            predicted_polygons = np.load(polygons_path)[args['name']]
        else:
            predicted_polygons = None
        if gt_polygons_path:
            gt_polygons = np.load(gt_polygons_path)[args['name']]
        else:
            gt_polygons = None

        visualize(args['name'], 
                  args['visual_out'], 
                  polygons=predicted_polygons,
                  gt_polygons=gt_polygons)
    else:
        tp = args['transforms_path']
        if tp.lower() == 'none':
            tp = None
        review(args['root_dir'],
               args['visual_out'],
               transforms_path=tp,
               allow_errors=args['allow_errors'],
               polygons_path=polygons_path,
               gt_polygons_path=gt_polygons_path,
               skip_true_negatives=args['skip_true_negatives'])
