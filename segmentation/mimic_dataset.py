"""Tools for restructuring your dataset to look like a common dataset.

The idea here is that deep learning repositories are often set up in a
way that make results (based on some common dataset) easily 
reproducible.  Training on your own dataset is own not always so 
straight-forward.

Datasets currently supported:
  * PASCAL VOC 2012


Usage Example:
IMG=/home/andy/Desktop/yts/deeplab_merged_results/fixed/images
SEG=/home/andy/Desktop/yts/deeplab_merged_results/fixed/segmentations
OUT=/home/andy/Desktop/yts/deeplab_merged_results/fixed
python mimic_dataset.py -i ${IMG} -m ${SEG} -o ${OUT}

IMG=os.path.expandvars('~/Desktop/yts/all-yts-images')
SEG=os.path.expandvars('~/Desktop/yts/yts_superclipped/segmentations')
OUT=os.path.expandvars('~/Desktop/yts')
run mimic_dataset.py -i IMG -m SEG -o OUT
"""
from __future__ import print_function
import os
import shutil
import cv2 as cv
import numpy as np


def mkdir(*vargs):
    os.mkdir(os.path.join(*vargs))


def is_jpeg(fn, extensions=('.jpg', '.jpeg')):
    return os.path.splitext(fn)[-1].lower() in extensions


def bounding_box(where, image_size):
    """Returns the minimal up-right bounding box extrema.
    Expects input formatted as if from an `np.where()`."""
    x, y, w, h = cv.boundingRect(np.stack(where).transpose()[:, None, :])
    center = np.array([x, y])
    size = np.array([w, h])
    xmin, ymin = np.round(center - size/2).astype('int')
    xmin, ymin = max(xmin, 0), max(ymin, 0)
    xmax, ymax = np.round(center + size/2).astype('int')
    xmax, ymax = min(xmax, image_size[0]), min(ymax, image_size[1])
    return xmin, ymin, xmax, ymax


def make_image_sets(image_dir, out_dir, ext='.png', train_portion=.8):
    images = [f[:-len(ext)] for f in os.listdir(image_dir) if f.endswith(ext)]
    np.random.shuffle(images)
    n = int(len(images) * train_portion)
    train_set = '\n'.join(images[:n])
    val_set = '\n'.join(images[:n])
    with open(os.path.join(out_dir, 'train.txt'), 'w+') as txt:
        txt.write(train_set)
    with open(os.path.join(out_dir, 'trainval.txt'), 'w+') as txt:
        txt.write(train_set + '\n' + val_set)
    with open(os.path.join(out_dir, 'val.txt'), 'w+') as txt:
        txt.write(val_set)


def create_voc_annotation(instance_seg_map, labels, name, filename,
                          tag='Unspecified', image_depth=3, pose='Unspecified',
                          truncated=0, difficult=0):
    """Expects a 2D image of pixel-wise labels.
    Assumes one object per class per image."""
    h, w = instance_seg_map.shape[:2]
    a = ("<annotation>\n"
         "\t<folder>VOC2012</folder>\n"
         "\t<filename>%s.jpg</filename>\n"
         "\t<source>\n"
         "\t\t<database>The VOC2007 Database</database>\n"
         "\t\t<annotation>PASCAL VOC2007</annotation>\n"
         "\t\t<image>%s</image>\n"
         "\t</source>\n"
         "\t<size>\n"
         "\t\t<width>%s</width>\n"
         "\t\t<height>%s</height>\n"
         "\t\t<depth>%s</depth>\n"
         "\t</size>\n"
         "\t<segmented>1</segmented>\n"  # means there exists a corr. seg map?
         "" % (name, tag, w, h, image_depth))

    for val in np.unique(instance_seg_map)[1:]:
        label = labels[val]
        xmin, ymin, xmax, ymax = \
            bounding_box(np.where(instance_seg_map == val), (w, h))
        a += ("\t<object>\n"
              "\t\t<name>%s</name>\n"
              "\t\t<pose>%s</pose>\n"
              "\t\t<truncated>%s</truncated>\n"
              "\t\t<difficult>%s</difficult>\n"
              "\t\t<bndbox>\n"
              "\t\t\t<xmin>%s</xmin>\n"
              "\t\t\t<ymin>%s</ymin>\n"
              "\t\t\t<xmax>%s</xmax>\n"
              "\t\t\t<ymax>%s</ymax>\n"
              "\t\t</bndbox>\n"
              "\t</object>\n"
              "" % (label, pose, truncated, difficult, xmin, ymin, xmax, ymax))
    a += "</annotation>\n"
    if filename is not None:
        with open(filename, 'w+') as f:
            f.write(a)
    return a


# credit: tensorflow/models/research/deeplab
def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
      A colormap for visualizing segmentation results.
    """
    def bit_get(val, idx):
        """Gets the bit value.

        Args:
          val: Input value, int or numpy int array.
          idx: Which bit of the input val.

        Returns:
          The "idx"-th bit of input val.
        """
        return (val >> idx) & 1
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap


def mimic_voc2012(image_dir, mask_dir, out_dir, labels, train_portion=0.8,
                  print_progress=True, convert_to_binary=False,
                  mask_prefix=''):
    """Restructures a dataset to mimic the PASCAL VOC 2012 dataset.

    Given a collection of jpeg images and corresponding k-nary
    semantic segmentation masks, creates a directory identical in
    structure to the PASCAL VOC 2012 dataset.
    The idea here is that deep learning repositories are often set up
    in a way that make results (based on some common dataset) easily
    reproducible.  Training on your own dataset is own not always so
    straight-forward.
    PASCAL VOC is one of the most common semantic segmentation datasets.

    Args:
        image_dir (str): a directory of jpeg images
        mask_dir (str): a directory of png grayscale segmentations masks
            named identically to those in `image_dir`.  Each
        out_dir (str): where to store the restructured dataset
        labels (str): A list of names corresponding to the unique values
            found in the masks.  Note that 0 will always be assume as
            background and 255 will be assumed as uncertain/boundary.
            E.g. `labels=['dog', 'cat'] means you used a 1 to signify a
            pixel is a dog, and a 2 to signify a pixel is a cat.
    The resulting directory structure will be identical to that from
    the 2012 downloadable, "VOCdevkit_18-May-2011.tar".
      + pascal_voc_seg
        + VOCdevkit
          + VOC2012
            + Annotations    -- xml annotations styled as in VOC
            + ImageSets
              + Action    -- N/A, for action-labels (this will be empty)
              + Layout    -- N/A, for ??? (this will be empty)
              + Main    -- for classifier-only ground-truth
              + Segmentation
            + JPEGImages    -- the images, unchanged
            + SegmentationClass    -- png segmentation masks (by class)
            + SegmentationObject    -- png segmentation masks (by instance)

    Other Important Notes about the VOC dataset:
    * "Bordering regions are marked with a 'void' label (index 255),
    indicating that the contained pixels can be any class including
    background."

    For further explanation of the PASCAL VOC 2012 dataset, see the
    "devkit_doc.pdf" available (inside devkit) on the official website.
    """
    mkdir(out_dir, 'pascal_voc_seg')
    mkdir(out_dir, 'pascal_voc_seg', 'VOCdevkit')
    voc2012 = os.path.join(out_dir, 'pascal_voc_seg', 'VOCdevkit', 'VOC2012')
    mkdir(voc2012)
    mkdir(voc2012, 'Annotations')
    mkdir(voc2012, 'ImageSets')
    mkdir(voc2012, 'ImageSets', 'Action')
    mkdir(voc2012, 'ImageSets', 'Layout')
    mkdir(voc2012, 'ImageSets', 'Main')
    mkdir(voc2012, 'ImageSets', 'Segmentation')
    mkdir(voc2012, 'JPEGImages')
    mkdir(voc2012, 'SegmentationClass')
    mkdir(voc2012, 'SegmentationObject')

    colormap = np.flip(create_pascal_label_colormap(), 1)
    for fn in os.listdir(image_dir):
        if print_progress:
            print('Working on', fn, '...', end='')
        name = os.path.splitext(fn)[0]
        full_mask_fn = os.path.join(mask_dir, mask_prefix + name + '.png')
        full_image_fn = os.path.join(image_dir, fn)

        image = cv.imread(full_image_fn)
        assert image is not None, "Failed to read image %s" % full_image_fn
        out_jpeg_name = os.path.join(voc2012, 'JPEGImages', name + '.jpg')
        if is_jpeg(fn):
            shutil.copyfile(full_image_fn, image)
        else:
            cv.imwrite(out_jpeg_name, image)

        mask = cv.imread(full_mask_fn)
        assert mask is not None, "No mask found with name %s" % full_mask_fn
        if convert_to_binary:
            if mask.ndim == 3:
                mask = mask.astype('int').sum(axis=2)
            mask = (mask > 0) * 1

        # Create xml annotation file
        xml_name = os.path.join(voc2012, 'Annotations', name + '.xml')
        create_voc_annotation(mask, labels, name, xml_name)

        # Create class-level segmentation mask
        cv.imwrite(os.path.join(voc2012, 'SegmentationClass', name + '.png'),
                   colormap[mask])
        if print_progress:
            print('Done.')

    if train_portion is not None:
        mask_dir = os.path.join(voc2012, 'SegmentationClass')
        lists_dir = os.path.join(voc2012, 'ImageSets', 'Segmentation')
        make_image_sets(mask_dir, lists_dir, '.png', train_portion)


if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--image_dir", required=True,
                      help="directory containing original JPEG images")
    args.add_argument("-m", "--mask_dir", required=True,
                      help="directory containing segmentation masks")
    args.add_argument("-o", "--out_dir", required=True,
                      help="where to store restructured dataset")
    args.add_argument("-p", "--train_portion", default=0.8,
                      help="portion of the image set to use for training")
    args.add_argument("-l", '--labels', default=None,
                       help='txt file listing label names (background '
                            'should be first)')
    args.add_argument("-b", '--binary', action='store_true', default=False,
                       help='convert to mask to binary')
    args.add_argument('--no_progress', action='store_true', default=False,
                       help="don't print progress")
    args.add_argument('--mask_prefix', default='',
                       help="in case all mask filenames have some junk "
                            "prefixed on them")
    args = vars(args.parse_args())

    if args['labels'] is None:
        LABELS = ['background', 'fish']
        CONVERT_TO_BINARY = True
        MASK_PREFIX = '3_'
    else:
        with open(args['labels'], 'r') as labels_txt:
            LABELS = [row for row in labels_txt]
        CONVERT_TO_BINARY = args['binary']
        MASK_PREFIX = args['mask_prefix']

    mimic_voc2012(args['image_dir'], args['mask_dir'], args['out_dir'],
                  LABELS, not args['no_progress'], CONVERT_TO_BINARY,
                  MASK_PREFIX)
