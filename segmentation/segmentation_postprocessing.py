#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Improve image segmentations images with various techniques.

Requirements
------------
* NumPy
* OpenCV (tested with version 3.4.1)
* scikit-image (tested with version 0.10.1)

Usage
-----
* To run a single image::

    $ python segmentation_postprocessing.py ???


* To run on all images in a directory::

    $ python segmentation_postprocessing.py ???


* For more options::

    $ python segmentation_postprocessing.py ???

...whoops .. forgot to write this.  See the CLI parsing at the bottom.
"""


from __future__ import division, print_function
import cv2 as cv  # v3.4.1
import numpy as np
import os
from skimage.segmentation import slic, mark_boundaries  # v0.10.1

try:
    from misc import Timer
except:
    from .misc import Timer


DEBUG = False


def remove_tail(seg_map, filter_width=25, margin=25, debug=False):
    """ Remove the fish's tail.

    Returns segmentation map with tail cut off by a vertical slice.
    Works by smoothing then using a concavity check of sorts.

     UNFINISHED -- CURRENTLY THIS METHOD DOESN'T WORK WELL.
     """
    h, w = seg_map.shape
    heights = []
    for col in seg_map.transpose():
        ones = np.where(col)[0]
        if len(ones):
            heights.append(ones.argmax() - ones.argmin())
        else:
            heights.append(0)
    smoothed = np.convolve(heights, np.ones(filter_width), 'same')
    gradient = np.convolve(np.gradient(smoothed), filter_width)

    nonzero_values = np.nonzero(gradient)[0]
    first_nonzero = nonzero_values.min()
    last_nonzero = nonzero_values.max()

    s = sorted(list(zip(np.abs(gradient), range(w))
                    )[first_nonzero + margin: last_nonzero - margin])
    _, ss = zip(*s[:3])
    tail_x = sorted(ss)[1]

    if debug:
        print(s[:10])
        print(ss)
        print(tail_x)
        print("adfasdf")
        print(first_nonzero)
        print(last_nonzero)
        import matplotlib.pyplot as plt
        plt.subplot(2, 1, 1)
        plt.plot(range(w), smoothed)
        plt.subplot(2, 1, 2)
        plt.plot(range(w)[first_nonzero + margin: last_nonzero - margin],
                 abs(gradient)[first_nonzero + margin: last_nonzero - margin])
        plt.show()
    return np.hstack((seg_map[:, :tail_x], np.zeros((h, w - tail_x)))
                     ).astype('uint8')


def colorize_superpixels(superpixel_mask, write_as=None):
    """Fills in each superpixel with a random color for visualization."""
    colors = np.array([np.random.rand(3) * 255 for _ in np.unique(superpixel_mask)])
    colorized = colors[superpixel_mask, :]
    if write_as is not None:
        cv.imwrite(write_as, colorized)
    return colorized


def threshold_over_superpixels(image, pixelwise_labels, n_superpixels,
                               threshold, debug=DEBUG, timer_enable=False):
    """Standard thresholding algorithm applied to superpixels.

    Pixels inside superpixels with mean intensity greater than or equal
    to `threshold` will be filled in with the value 255.  All other
    pixels will be 0."""

    assert pixelwise_labels.ndim == 2
    pixelwise_labels = pixelwise_labels.astype('bool')

    with Timer("Finding superpixels", enable=timer_enable):
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore')
            superpixel_mask = slic(image, n_superpixels, sigma=5).astype('uint32')
    n_superpixels = superpixel_mask.max() + 1  # may differ from requested
    if debug:
        marked = mark_boundaries(image, superpixel_mask) * 255
        marked = marked + 50*np.stack([pixelwise_labels]*3, 2)
        cv.imwrite('dbg_superpixels_image.png', marked)
        marked = mark_boundaries(pixelwise_labels, superpixel_mask) * 255
        cv.imwrite('dbg_superpixels_mask.png', marked)
        with Timer("Colorize", enable=timer_enable):
            colorize_superpixels(superpixel_mask, 'dbg_superpixels.png')

    with Timer("Finding superpixel labels", enable=timer_enable):
        labels, counts = np.unique((1 + superpixel_mask) * pixelwise_labels,
                                   return_counts=True)
        positive_votes = np.zeros(n_superpixels, dtype='int64')

        if threshold == 0:
            positive_votes[labels[1:] - 1] = 1
            superpixel_labels = positive_votes
        else:
            positive_votes[labels[1:] - 1] = counts[1:]
            _, superpixel_sizes = np.unique(superpixel_mask, return_counts=True)
            superpixel_labels = (positive_votes/superpixel_sizes) >= threshold

    with Timer("Finding new superpixel-wise segmentation map", enable=timer_enable):
        new_seg_map = superpixel_labels[superpixel_mask]
    return new_seg_map.astype('uint8') * 255


def compose_affine_transforms(M2, M1):
    A1, b1 = M1[:, :2], M1[:, 2].reshape(2, 1)
    A2, b2 = M2[:, :2], M2[:, 2].reshape(2, 1)
    return np.hstack((np.dot(A2, A1), np.dot(A2, b1) + b2))


def draw_rrect(image, rrect, color=(255, 0, 0)):
    pts = cv.boxPoints(rrect)
    for i in range(len(pts)):
        pt1 = tuple(pts[i - 1].astype('int')[::-1])
        pt2 = tuple(pts[i].astype('int')[::-1])
        cv.line(image, pt1, pt2, color, thickness=5)


def bbox_angle(rrect):
    """An ad-hoc fix for a getting the angle of the minimal bounding box."""
    if rrect[2] < -60:
        pts = cv.boxPoints(rrect)
        l1 = pts[0] - pts[1]
        l2 = pts[0] - pts[-1]
        if np.linalg.norm(l1) > np.linalg.norm(l2):
            cosine = np.dot(l1, [0, 1]) / np.linalg.norm(l1)
        else:
            cosine = np.dot(l2, [0, 1]) / np.linalg.norm(l2)
        return np.degrees(np.arccos(cosine))
    return rrect[2]


def bounding_box(where):
    """Returns the minimal bounding box in form (center, size, angle).
    Expects input formatted as if from an `np.where()`."""
    return cv.minAreaRect(np.stack(where).transpose()[:, None, :])


def only_largest_connected_component(seg_map, debug=False):
    retval, labels = cv.connectedComponents(seg_map)
    fishy_lbl = 1 + np.array([(labels == i).sum()
                              for i in range(1, labels.max() + 1)]).argmax()
    if debug:
        cv.imwrite('dbg_2a_connected_components.png', labels/labels.max()*255)
    return (labels == fishy_lbl) * 255


# https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
def fill_holes(image, invert=True, debug=False):
    if invert:
        image = (image == 0) * 255
    th, im_th = cv.threshold(image.astype('uint8'), 220, 255,
                              cv.THRESH_BINARY_INV)
    im_floodfill = im_th.copy()
    if debug:
        cv.imwrite('dbg_fill1.png', im_th)
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    _, im_floodfill, _, _ = cv.floodFill(im_floodfill, mask, (0, 0), 255)
    if debug:
        cv.imwrite('dbg_fill2.png', im_floodfill)
    im_floodfill_inv = cv.bitwise_not(im_floodfill)
    if debug:
        cv.imwrite('dbg_fill3a.png', im_floodfill_inv)
        cv.imwrite('dbg_fill3b.png', im_th)
    return im_th | im_floodfill_inv


def process_segmentation(seg_map, no_transforms=False, scale=.75,
                         tailless=False, initial_M=None, invert=False, 
                         initial_scaling=0.5, debug=DEBUG):
    """Does the following post-processing steps (on a binary segmentation map):
        1. eliminate all but largest connected component
        2. fill in holes
        3. center fish (on centroid)
        4. rotate to align the minimal bounding box with coordinate axes
        5. scale image so width of minimal bounding box is `scale` times the
        image width
    """
    rows, cols = seg_map.shape

    if initial_M is None:
        cumM = np.hstack((np.identity(2), np.zeros((2, 1))))
    else:
        cumM = initial_M

    if invert:
        seg_map = (np.logical_not(seg_map) * 255).astype('uint8')

    if debug:
        original = seg_map.copy()

    # scale initially so that parts aren't cut off by future future 
    # transforms (e.g. tail tips aren't cut off when centering)
    if initial_scaling != 1 and not no_transforms:
        A = initial_scaling * np.identity(2)
        x0 = (np.array((cols, rows))/2.).reshape(2, 1)
        M = np.hstack((A, x0 - np.dot(A, x0)))
        seg_map = cv.warpAffine(seg_map // 255, M, (cols, rows)) * 255
        cumM = compose_affine_transforms(M, cumM)

    if debug:
        comparison = np.stack((np.zeros((rows, cols)), original, seg_map), 2)
        cv.imwrite('dbg_1_initial_scaling.png', comparison)
        previous = seg_map.copy()

    # eliminate all but largest connected component
    seg_map = only_largest_connected_component(seg_map, debug=debug)

    # fill in holes
    seg_map = fill_holes(seg_map)

    if debug:
        comparison = np.stack((np.zeros((rows, cols)), original, seg_map), 2)
        cv.imwrite('dbg_2b_holes_filled.png', comparison)
        previous = seg_map.copy()

    if no_transforms:
        return seg_map, None

    # center fish
    m = cv.moments(seg_map, binaryImage=True)
    centroid = np.array((m['m10']/m['m00'], m['m01']/m['m00']))
    shift = np.array((cols, rows))/2 - centroid
    M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    seg_map = cv.warpAffine(seg_map // 255, M, (cols, rows)) * 255
    cumM = compose_affine_transforms(M, cumM)

    if debug:
        comparison = np.stack((np.zeros((rows, cols)), previous, seg_map), 2)
        cv.circle(comparison, tuple(centroid.astype('int')), 10, (255, 0, 0), -1)
        cv.circle(comparison, (cols//2, rows//2), 10, (255, 255, 255), -1)
        cv.imwrite('dbg_3_centered.png', comparison)
        previous = seg_map.copy()

    # align shortest radial through centroid with vertical axis
    rrect = bounding_box(np.where(seg_map))
    deg = bbox_angle(rrect)
    M = cv.getRotationMatrix2D((cols/2, rows/2), -deg, 1)
    seg_map = cv.warpAffine(seg_map // 255, M, (cols, rows)) * 255
    cumM = compose_affine_transforms(M, cumM)

    if debug:
        comparison = np.stack((np.zeros((rows, cols)), previous, seg_map), 2)
        draw_rrect(comparison, rrect, (0, 255, 0))
        rrect = bounding_box(np.where(seg_map))
        draw_rrect(comparison, rrect, (0, 0, 255))
        cv.imwrite('dbg_4_rotated.png', comparison)
        previous = seg_map.copy()
        print(M)
        x0 = np.array((cols/2, rows/2)).reshape(2, 1)
        print(x0 + np.dot(M[:, :2], -x0))

    # scale image so width of minimal bounding box is always 90% image width
    _, size, _ = bounding_box(np.where(seg_map))
    A = scale*cols/max(size) * np.identity(2)
    x0 = (np.array((cols, rows))/2.).reshape(2, 1)
    M = np.hstack((A, x0 - np.dot(A, x0)))
    seg_map = cv.warpAffine(seg_map // 255, M, (cols, rows)) * 255
    cumM = compose_affine_transforms(M, cumM)

    if debug:
        comparison = np.stack((np.zeros((rows, cols)), previous, seg_map), 2)
        draw_rrect(comparison, rrect)
        cv.imwrite('dbg_5_scaled.png', comparison)
        comparison = np.stack((np.zeros((rows, cols)), original, seg_map), 2)
        cv.imwrite('dbg_6_comparison.png', comparison)
        _, size, _ = bounding_box(np.where(seg_map))
        if not np.isclose(scale, max(size)/cols):
            print("something is wrong with scaling %s != %s"
                  "" % (scale, max(size)/cols))

    if tailless:
        seg_map = remove_tail(seg_map)
        if debug:
            cv.imwrite('dbg_7_tailless.png', seg_map)
        return process_segmentation(seg_map, no_transforms=no_transforms,
                                    scale=scale, tailless=False,
                                    initial_M=cumM, debug=DEBUG)
    return seg_map, cumM


def process_all(image_dir, out_dir):
    for fn in os.listdir(image_dir):
        full_fn = os.path.join(image_dir, fn)

        affine_transforms = {}
        if fn.lower().endswith('.png'):
            image = cv.imread(full_fn)
            seg_map, transform = process_segmentation(image)
            affine_transforms[os.path.splitext(fn)[0]] = transform
            cv.imwrite(os.path.join(out_dir, fn), seg_map)
        np.savez(os.path.join(out_dir, 'affine_transforms.npz'),
                 **affine_transforms)


def clip_image(image, seg_map, no_transforms=False, tailless=False,
               n_superpixels=5000, superpixel_threshold=0.5, debug=DEBUG):
    """ Applies superpixel thresholding to improve segmentations.

    Parameters
    ----------
    image : np.array
        The original image
    seg_map : np.array
        The binary segmentation map to be improved
    no_transforms : bool
        If false, no affine transformations (e.g. centering, alignment,
        scaling) will be performed in post-processing.
    tailless : bool
        (EXPERIMENTAL) If true, the fish's tail will be removed.
    n_superpixels : int or list of ints
        The number of superpixels to use.  If a list, superpixel
        thresholding will be performed multiple times, iteratively.
    superpixel_threshold : 0 <= number <= 1
        The threshold to use
    debug : bool
        If true, intermediate results will be written to help visualize
        and debug the algorithm.

    Returns
    -------
    np.array
        The original `image` after being transformed by `aff`
    np.array
        The (hopefully improved) segmentation map.
    np.array
        The affine transformation used in post-processing.

    Notes
    -----
    * If the `seg_map` is taller than `image` (as is
    expected with segmentations created by "deeplab v3+"), the bottom
    will be cropped off."""
    seg_map = ((seg_map > 0) * 255).astype('uint8')
    image = np.asarray(image)  # in case `image` is a PIL object

    if seg_map.ndim == 3:
            seg_map = seg_map[:, :, 0]

    # scale and crop seg_map to match original image
    h, w = image.shape[:2]
    if w > seg_map.shape[1]:
        full_sized_seg_map = cv.resize(seg_map, (w, w))[:h]
    else:
        full_sized_seg_map = seg_map[:h]

    # improve seg_map by making superpixels vote together
    if isinstance(n_superpixels, int):
        n_superpixels = [n_superpixels]
    for n in n_superpixels:
        full_sized_seg_map = threshold_over_superpixels(
            image, full_sized_seg_map, n, superpixel_threshold, debug=debug)
        if debug:
            cv.imwrite('dbg_superthresh_%s.png' % n, full_sized_seg_map)

    # process segmentation and (maybe) transform original to match it
    processed_seg_map, aff = \
        process_segmentation(full_sized_seg_map, no_transforms=no_transforms,
                             tailless=tailless, debug=debug)
    if not no_transforms:
        image = cv.warpAffine(image, aff, (w, h))

    return image, processed_seg_map, aff


def is_image(fn, extensions=('jpeg', 'jpg', 'png')):
    return fn.split('.')[-1].lower() in extensions


def clip_all(originals_dir, segmentations_dir, out_dir, no_transforms=False,
             prefix='', tailless=False, n_superpixels=5000,
             superpixel_threshold=1, debug=DEBUG):
    """ Applies superpixel thresholding to improve segmentations.

    Assumes all originals are stored in single directory and all
    segmentations are stored in a second directory under that same name,
    but with  ".png" file.

    Note: Post-processing will be applied after superpixel thresholding.
    A dictionary containing the affine transforms used will be saved
    as "affine_transforms.npz" in `output_dir`.  To open this use
    `np.load`.

    See `clip_image` docstring for more info."""

    if not os.path.isdir(os.path.join(out_dir, 'segmentations')):
        os.mkdir(os.path.join(out_dir, 'segmentations'))
    if not os.path.isdir(os.path.join(out_dir, 'images')):
        os.mkdir(os.path.join(out_dir, 'images'))
    if not os.path.isdir(os.path.join(out_dir, 'clipped')):
        os.mkdir(os.path.join(out_dir, 'overlays'))

    affine_transforms = {}
    for fn in os.listdir(originals_dir):
        if not is_image(fn):
            continue
        print('Working on %s ... ' % fn, end='')
        try:
            # load orginal and segmentation image files
            name = os.path.splitext(fn)[0]
            original_path = os.path.join(imdir, fn)
            seg_fn = prefix + name + '.png'
            seg_path = os.path.join(segmentations_dir, seg_fn)
            image, seg_map = cv.imread(original_path), cv.imread(seg_path)

            if image is None:
                raise Exception('%s could not be read.' % original_path)
            if seg_map is None:
                raise Exception('%s could not be read.' % seg_path)

            if seg_map.ndim == 3:
                seg_map = np.split(seg_map, 3, axis=2)[0]

            # process and save results
            new_image, new_seg_map, mat = clip_image(
                image, seg_map, no_transforms, tailless=tailless,
                n_superpixels=n_superpixels,
                superpixel_threshold=superpixel_threshold, debug=debug)
            # cv.imwrite(os.path.join(outdir, 'overlays', seg_fn), overlay)
            # cv.imwrite(os.path.join(outdir, 'images', seg_fn), new_image)
            cv.imwrite(os.path.join(out_dir, 'segmentations', seg_fn),
                       (new_seg_map * 255).astype('uint8'))
            affine_transforms[name] = mat

        except Exception as e:
            print('ERROR:', e)
            continue
            # raise

        print('Done.')
    np.savez(os.path.join(out_dir, 'affine_transforms.npz'), **affine_transforms)


if __name__ == '__main__':
    from sys import argv
    import shutil
    from review_segmentations import overlay
    if len(argv) == 2 and argv[1] == 'test':
        test = '/home/andy/Desktop/yts_segmentations/3_ABJ4.png'
        test_image = cv.imread(test, 0)
        test_result, m = process_segmentation(test_image, debug=True)
        cv.imwrite('test_result.png', test_result)
    if len(argv) == 2 and argv[1] == 'testclip':
        seg_map_ = cv.imread('/home/andy/Desktop/yts_segmentations/3_AB1.png')
        seg_map_ = seg_map_[:, :, 0]
        image_ = cv.imread('/home/andy/Desktop/all-yts-images/AB1.JPG')
        with Timer("Clipping Image"):
            new_image_, new_seg_map_, affine_transform = \
                clip_image(image_, seg_map_,
                           n_superpixels=5000,
                           superpixel_threshold=1)
        cv.imwrite('test_result.png', overlay(new_image_, seg_map_))
    elif len(argv) == 2 and argv[1] == 'runall':
        imdir = '/home/andy/Desktop/yts_segmentations'
        outdir = '/home/andy/Desktop/yts_segmentations_processed'
        process_all(imdir, outdir)
    elif len(argv) == 2 and argv[1] == 'clipall':
        imdir = '/home/andy/Desktop/yts/deeplab_merged_results/images'
        segs_dir = '/home/andy/Desktop/yts/deeplab_merged_results/segmentations'
        outdir = '/home/andy/Desktop/yts/deeplab_merged_results/fixed'
        os.mkdir(outdir)
        shutil.copyfile('visualize.py',
                        os.path.join(outdir, 'visualize.py'))
        clip_all(imdir, segs_dir, outdir, prefix='')
    elif len(argv) == 6 and argv[1] == 'clipthis':
        imdir = argv[2]
        segs_dir = argv[3]
        outdir = argv[4]
        prefix = argv[5]
        os.mkdir(outdir)
        shutil.copyfile('visualize.py',
                        os.path.join(outdir, 'visualize.py'))
        clip_all(imdir, segs_dir, outdir, prefix=prefix)
    elif len(argv) == 2 and argv[1] == 'clipallquick':
        imdir = '/home/andy/Desktop/all-yts-images'
        segs_dir = '/home/andy/Desktop/yts_segmentations'
        outdir = '/home/andy/Desktop/yts_quick_segged'
        clip_all(imdir, segs_dir, outdir, prefix='3_', n_superpixels=[])
    elif len(argv) == 2 and argv[1] == 'clipalltails':
        imdir = '/home/andy/Desktop/all-yts-images'
        segs_dir = '/home/andy/Desktop/yts_segmentations'
        outdir = '/home/andy/Desktop/yts_clipped_tailless'
        clip_all(imdir, segs_dir, outdir, prefix='3_', tailless=True)
    else:
        print('CLI arguments not understood.\n  argv =', argv)
