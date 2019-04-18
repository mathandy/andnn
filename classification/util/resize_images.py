import cv2 as cv

try:
    from .util import (is_jpeg_or_png, Timer, resize_preserving_aspect_ratio,
                       walk_and_transform_files_in_place)
except:
    from util import (is_jpeg_or_png, Timer, resize_preserving_aspect_ratio,
                      walk_and_transform_files_in_place)


def resize_images_in_place(root_dir, transform_fcn, filter_fcn=is_jpeg_or_png,
                           raise_exceptions=False, remove_problem_files=True):
    walk_and_transform_files_in_place(
        root_dir=root_dir, transform_fcn=transform_fcn, filter_fcn=filter_fcn,
        raise_exceptions=raise_exceptions,
        remove_problem_files=remove_problem_files,)


transforms = ['no_warp_resize']
filter_dict = {'images': is_jpeg_or_png}

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
