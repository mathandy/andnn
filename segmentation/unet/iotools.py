"""
Credit: (Feb-2019) Some code here inspired or modified from
https://github.com/zhixuhao/unet
"""
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from segmentation_models.backbones import get_preprocessing
import cv2 as cv


# try:
#     from andnn.finetune import preloader
#     from andnn.utils import is_jpeg_or_png
# except:
#     from .andnn.finetune import preloader
#     from .andnn.utils import is_jpeg_or_png
from util import resize_preserving_aspect_ratio as resize


def is_jpeg_or_png(fn):
    return os.path.splitext(fn)[1][1:].lower() in ('jpg', 'jpeg', 'png')


def crop(image, mask):
    h, w = image.shape[:2]
    rows, cols = np.random.randint(h // 4, h), np.random.randint(w // 4, w)
    i0, j0 = np.random.randint(0, h - rows), np.random.randint(0, w - cols)
    return {'image': image[i0:i0 + rows, j0:j0 + cols],
            'mask': mask[i0:i0 + rows, j0:j0 + cols]}


def data_generator(data_dir, batch_size, input_size=None,
                   keras_augmentations=None, preprocessing_function_x=None,
                   preprocessing_function_y=None, preload=False,
                   cached_preloading=False, verbosity=True, mode=None,
                   random_crops=0.0):
    if keras_augmentations is None:
        keras_augmentations = dict()

    if random_crops:
        print('\n\nWARNING: Crop size hard coded as 25%-100% of image size.\n\n')

    # mask and image generators must use same seed
    keras_seed = np.random.randint(314)

    # note: preprocessing_fcn will run after image is resized and augmented
    image_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function_x,
        **keras_augmentations)
    mask_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function_y,
        **keras_augmentations)

    if preload:
        # from tempfile import tempdir
        # x = preloader(dataset_dir=data_dir,
        #               img_shape=tuple(list(input_size)[::-1] + [3]),
        #               subset='images',
        #               cached_preloading=cached_preloading,
        #               storage_dir=os.path.join(tempdir, 'unet-preloader'),
        #               verbose=verbosity)
        # y = preloader(dataset_dir=data_dir,
        #               img_shape=tuple(list(input_size)[::-1] + [3]),
        #               subset='segmentations',
        #               cached_preloading=cached_preloading,
        #               storage_dir=os.path.join(tempdir, 'unet-preloader'),
        #               verbose=verbosity)
        #
        # pair_generator = image_datagen.flow(
        #     x=(x, y),
        #     y=None,
        #     batch_size=32,
        #     shuffle=True,
        #     seed=keras_seed,
        #     subset=mode)
        raise NotImplementedError
    else:
        image_generator = image_datagen.flow_from_directory(
            directory=data_dir,
            classes=['images'],
            class_mode=None,
            color_mode='rgb',
            target_size=input_size,
            batch_size=batch_size,
            seed=keras_seed,
            subset=mode)

        mask_generator = mask_datagen.flow_from_directory(
            directory=data_dir,
            classes=['segmentations'],
            class_mode=None,
            color_mode='grayscale',
            target_size=input_size,
            batch_size=batch_size,
            seed=keras_seed,
            subset=mode)

        pair_generator = zip(image_generator, mask_generator)

    for (unaugmented_image_batch, unaugmented_mask_batch) in pair_generator:

        if random_crops:
            image_batch = np.empty((batch_size,) + input_size[::-1] +
                                   unaugmented_image_batch.shape[3:])
            # mask_batch = np.empty((batch_size,) + input_size[::-1] + mb.shape[3:])
            mask_batch = np.empty((batch_size,) + input_size[::-1] + (1,))
            for k in range(batch_size):
                cropped = crop(**{'image': unaugmented_image_batch[k],
                                  'mask': unaugmented_mask_batch[k]})
                if (cropped['image'].shape[0] != input_size[1] or
                    cropped['image'].shape[1] != input_size[0]):
                    image_batch[k] = resize(cropped['image'], dsize=input_size)
                    mask_batch[k] = resize(np.squeeze(cropped['mask']), dsize=input_size)[:, :, None]
                else:
                    image_batch[k], mask_batch[k] = cropped['image'], cropped['mask']
        else:
            image_batch, mask_batch = unaugmented_image_batch, unaugmented_mask_batch

        image_batch = image_batch.astype('float32')/255
        mask_batch = mask_batch.astype('float32')/255
        # print('\nimage', image_batch.shape, image_batch.dtype, image_batch.min(), image_batch.max())
        # print('mask', mask_batch.shape, mask_batch.dtype, mask_batch.min(), mask_batch.max())
        yield image_batch, mask_batch


def get_data_generators(data_dir, backbone='resnet34', batch_size=2,
                        input_size=(256, 256), keras_augmentations=None,
                        preprocessing_function_x=None,
                        preprocessing_function_y=None,
                        preload=False, cached_preloading=False,
                        presplit=True, random_crops=False):
    if keras_augmentations is None:
        keras_augmentations = dict()
    if preprocessing_function_x is None:
        preprocessing_function_x = get_preprocessing(backbone)

    training_generator = \
        data_generator(data_dir=os.path.join(data_dir, 'train'),
                       batch_size=batch_size,
                       input_size=input_size,
                       keras_augmentations=keras_augmentations,
                       preprocessing_function_x=preprocessing_function_x,
                       preprocessing_function_y=preprocessing_function_y,
                       preload=preload, cached_preloading=cached_preloading,
                       mode=None if presplit else 'training',
                       random_crops=random_crops)

    validation_generator = \
        data_generator(data_dir=os.path.join(data_dir, 'val'),
                       batch_size=batch_size,
                       input_size=input_size,
                       keras_augmentations=None,
                       preprocessing_function_x=preprocessing_function_x,
                       preprocessing_function_y=preprocessing_function_y,
                       preload=preload, cached_preloading=cached_preloading,
                       mode=None if presplit else 'validation',
                       random_crops=False)

    def count_images(directory):
        images = [x for x in os.listdir(os.path.join(directory))
                  if is_jpeg_or_png(x)]
        return len(images)

    training_steps_per_epoch = count_images(
        os.path.join(data_dir, 'train', 'images')) / batch_size
    validation_steps_per_epoch = count_images(
        os.path.join(data_dir, 'val', 'images')) / batch_size
    return (training_generator, validation_generator,
            training_steps_per_epoch, validation_steps_per_epoch)
