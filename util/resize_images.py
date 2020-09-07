"""Walk through directory resizing images (in place), preserving aspect ratio.

See CLI options and supported border types below.

Note if you do not want to preserve aspect ratios, you might instead use
ImageMagick, e.g. do something like

```shell script
# for f in "$(find . -type f -iname "*.jpg")"; do
#     echo "$f" "$f"
#     convert "${f}" -resize 400x400\! "${f}"  # this doesn't seem to work
# done

for dir in */; do
    cd $dir
    for f in *.jpg; do
        convert "$f" -resize 400x400\! "$f"
    done
    cd ..
done
```
"""

import cv2 as cv
import numpy as np
import os
from sys import stdout
from time import time as current_time


BORDER_TYPES = {

    'replicate': cv.BORDER_REPLICATE,
    'reflect': cv.BORDER_REFLECT,
    'reflect101': cv.BORDER_REFLECT_101,
    'wrap': cv.BORDER_WRAP,
    'constant': cv.BORDER_CONSTANT
}


def is_jpeg_or_png(fn):
    return os.path.splitext(fn)[1][1:].lower() in ('jpg', 'jpeg', 'png')


transforms = ['no_warp_resize']
filter_dict = {'images': is_jpeg_or_png}


def pnumber(x, n=5, pad=' '):
    """Takes in a float, outputs a string of length n."""
    s = str(x)
    try:
        return s[:n]
    except IndexError:
        return pad*(n - len(s)) + s


class Timer:
    """A simple tool for timing code while keeping it pretty."""

    def __init__(self, mes='', pretty_time=True, n=4, pad=' ', enable=True):
        self.mes = mes  # append after `mes` + '...'
        self.pretty_time = pretty_time
        self.n = n
        self.pad = pad
        self.enabled = enable

    def format_time(self, et, n=4, pad=' '):
        if self.pretty_time:
            if et < 60:
                return '{} sec'.format(pnumber(et, n, pad))
            elif et < 3600:
                return '{} min'.format(pnumber(et / 60, n, pad))
            else:
                return '{} hrs'.format(pnumber(et / 3600, n, pad))
        else:
            return '{} sec'.format(et)

    def __enter__(self):
        if self.enabled:
            stdout.write(self.mes + '...')
            stdout.flush()
            self.t0 = current_time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.t1 = current_time()
        if self.enabled:
            print("done (in {})".format(
                self.format_time(self.t1 - self.t0, self.n, self.pad)))
            stdout.flush()


def rescale_by_width(image, target_width, method=cv.INTER_LANCZOS4):
    """Rescale `image` to `target_width` (preserving aspect ratio)."""
    h = int(round(target_width * image.shape[0] / image.shape[1]))
    return cv.resize(image, (target_width, h), interpolation=method)


def rescale_by_height(image, target_height, method=cv.INTER_LANCZOS4):
    """Rescale `image` to `target_height` (preserving aspect ratio)."""
    w = int(round(target_height * image.shape[1] / image.shape[0]))
    return cv.resize(image, (w, target_height), interpolation=method)


def resize_preserving_aspect_ratio(image, dsize, output=None, color=(0, 0, 0),
                                   interpolation=cv.INTER_LANCZOS4,
                                   border_type=cv.BORDER_CONSTANT,
                                   return_coordinate_transform=False):
    """Resize image, using padding if necessary to avoid warping.

    dsize = (w, h)

    Border Types
    ------------
    * 'replicate':     aaaaaa|abcdefgh|hhhhhhh
    * 'reflect':       fedcba|abcdefgh|hgfedcb
    * 'reflect101':   gfedcb|abcdefgh|gfedcba
    * 'wrap':          cdefgh|abcdefgh|abcdefg
    * 'constant':      iiiiii|abcdefgh|iiiiiii  with some specified 'i'

    See Also
    --------
    * transform_points_with_resize()
    """

    if isinstance(image, str):
        image = cv.imread(image)
    if isinstance(border_type, str):
        border_type = BORDER_TYPES[border_type]

    original_shape = image.shape
    image_apect_ratio = image.shape[0] / image.shape[1]
    desired_apect_ratio = dsize[1] / dsize[0]

    if image_apect_ratio > desired_apect_ratio:  # must pad to widen
        image = rescale_by_height(image, dsize[1], method=interpolation)
    elif image_apect_ratio < desired_apect_ratio:
        image = rescale_by_width(image, dsize[0], method=interpolation)
    else:
        image = cv.resize(src=image, dsize=dsize, interpolation=interpolation)

    dh, dw = dsize[1] - image.shape[0], dsize[0] - image.shape[1]
    top, bottom = dh//2, dh - (dh//2)
    left, right = dw//2, dw - (dw//2)

    scaled_shape = image.shape

    image = cv.copyMakeBorder(src=image,
                              top=top, bottom=bottom, left=left, right=right,
                              borderType=border_type,
                              value=color)

    ho, wo = original_shape[:2]
    hs, ws = scaled_shape[:2]
    transformation = np.array([[ws/wo, 0,     dw//2],
                               [0,     hs/ho, dh//2]])

    if output is not None:
        cv.imwrite(filename=output, img=image)
    if return_coordinate_transform:
        return image, transformation
    return image


def walk_and_transform_files_in_place(root_dir, transform_fcn,
                                      filter_fcn=None,
                                      raise_exceptions=True,
                                      remove_problem_files=False):
    """Walk through `root_dir` applying `transform_fcn` to each image."""
    for directory, _, files in os.walk(root_dir):
        if not files:
            continue
        relative_path = os.path.sep.join(directory.split(os.path.sep)[-2:])
        with Timer("Processing files in " + relative_path):
            for fn in files:
                fn_full = os.path.join(directory, fn)
                if not filter_fcn(fn_full):
                    continue
                try:
                    transform_fcn(fn_full, fn_full)
                except Exception as exception:
                    print("\nThe follwoing exception was encountered "
                          "processing '%s' (file will be removed):\n%s"
                          "" % (fn, exception))
                    if raise_exceptions:
                        raise
                    if remove_problem_files:
                        os.remove(fn_full)


def resize_images_in_place(root_dir, transform_fcn, filter_fcn=is_jpeg_or_png,
                           raise_exceptions=False, remove_problem_files=True):
    walk_and_transform_files_in_place(
        root_dir=root_dir, transform_fcn=transform_fcn, filter_fcn=filter_fcn,
        raise_exceptions=raise_exceptions,
        remove_problem_files=remove_problem_files,)


if __name__ == '__main__':
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument('root_dir',
                      help="Directory (possibly with subdirectories) "
                           "containing images")
    args.add_argument('size', nargs=2, type=int,
                      help="Desired width and height (only applicable if "
                           "resizing.")
    args.add_argument('-w', '--allow_warping',
                      default=False, action='store_true',
                      help="Do not use padding, allow warping.")
    args.add_argument('-f', '--filter', default='images',
                      help="What types of files to transform (defaults "
                           "to 'images', which means JPEG and PNG files).  "
                           "The options are:\n%s"
                           "" % '\n'.join(filter_dict.keys()))
    args.add_argument('-b', '--border_type',
                      default=cv.BORDER_CONSTANT,
                      help="Border type: 'replicate', 'reflect', "
                           "'reflect101', 'wrap', or 'constant'.")
    args.add_argument('-e', '--stop_on_exceptions',
                      default=False, action='store_true',
                      help="Report and continue if exception is thrown.")
    args.add_argument('-k', '--keep_exceptional_files',
                      default=False, action='store_true',
                      help="Keep files that are not transformed successfully.")
    args = args.parse_args()

    if args.allow_warping:
        def transform(fn_in, fn_out):
            cv.imwrite(fn_in, cv.resize(src=fn_out, dsize=tuple(args.size)))
    else:
        def transform(fn_in, fn_out):
            resize_preserving_aspect_ratio(
                image=fn_in, dsize=tuple(args.size), output=fn_out,
                border_type=args.border_type)

    walk_and_transform_files_in_place(
        root_dir=args.root_dir,
        transform_fcn=transform,
        filter_fcn=filter_dict[args.filter],
        raise_exceptions=args.stop_on_exceptions,
        remove_problem_files=not args.keep_exceptional_files)
