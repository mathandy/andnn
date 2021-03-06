"""
DS=~/Dropbox
INPUT=$DS/hand-segmented-fish-new-old-split-hiRes
python3 unet.py train $INPUT --augment -l $DS/unet-logs -k $DS/segmentation-model.h5 -n firstrun

DS=~/Dropbox
INPUT=$DS/hand-segmented-fish-new-old-split-hiRes
python3 unet.py train $INPUT --augment --show_training_data
"""
from __future__ import absolute_import, print_function, division
from keras.callbacks import (
    ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau, Callback)

from keras.models import load_model
from keras.optimizers import Adam
import numpy as np
import os
import cv2 as cv
from skimage.io import imread, imsave
from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.utils import set_trainable
from time import time
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from tempfile import tempdir
tempdir = tempdir if tempdir else '/tmp'


try:
    from .iotools import get_data_generators, resize
    from ..visualize import visualize
    from .util import is_image
    from ...custom_callbacks import VisualValidation, LRFinder
except:
    from custom_callbacks import VisualValidation, LRFinder
    from iotools import get_data_generators, resize
    from visualize import visualize
    from util import is_image


def get_callbacks(checkpoint_path=None, verbose=None, batch_size=None,
                  patience=None, logdir=None, run_name=None,
                  visual_validation_samples=None, steps_per_report=None,
                  input_size=None, predict_fcn=None, steps_per_epoch=None,
                  epochs=None):
    callbacks = dict()
    if checkpoint_path is not None:
        callbacks['ModelCheckpoint'] = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            verbose=int(verbose),
            save_best_only=True,
            save_weights_only=False,
            mode='auto',
            period=1)
    if patience is not None:
        callbacks['EarlyStopping'] = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=patience,
            verbose=int(verbose),
            mode='auto')
    # if patience is not None:
    #     callbacks['ReduceLROnPlateau'] = ReduceLROnPlateau(
    #         monitor='val_loss',
    #         factor=0.1,
    #         patience=patience,
    #         verbose=0,
    #         mode='auto',
    #         min_delta=0.0001,
    #         cooldown=0,
    #         min_lr=0)
    if steps_per_epoch is not None and epochs is not None:
        LRFinder(min_lr=1e-5,
                 max_lr=1e-2,
                 steps_per_epoch=steps_per_epoch,
                 epochs=epochs)
    if logdir is not None:
        # callbacks['TensorBoard'] = TensorBoard(
        #     log_dir=os.path.join(logdir, run_name),
        #     histogram_freq=0,
        #     batch_size=batch_size,
        #     write_graph=False,
        #     write_grads=False,
        #     write_images=False,
        #     embeddings_freq=0,
        #     embeddings_layer_names=None,
        #     embeddings_metadata=None,
        #     embeddings_data=None,
        #     update_freq=steps_per_report if steps_per_report else 'epoch')
        callbacks['TensorBoard'] = TensorBoard(
            log_dir=os.path.join(logdir, run_name))
    if visual_validation_samples is not None and predict_fcn is not None:
        callbacks['VisualValidation'] = VisualValidation(
            image_paths=visual_validation_samples,
            output_dir=os.path.join(logdir, run_name, 'visual_validation'),
            input_size=input_size,
            predict_fcn=predict_fcn)
    return callbacks


def show_generated_pairs(generator, fx=.1, fy=.1):
    for image_batch, mask_batch in generator:
        for image, mask in zip(image_batch, mask_batch):
            bgr = np.flip(image, axis=2)

            visual = visualize(bgr, mask)
            cv.imshow('', cv.resize(visual, None, fx=fx, fy=fy))
            print(image.shape, mask.shape)

            cv.waitKey(10)
            keypress = input()
            if keypress == 'q':
                cv.destroyAllWindows()
                return


def preprocess(image, input_size=None, backbone='resnet34', preprocessing_fcn=None):

    if isinstance(image, str):
        image = imread(image)

    if preprocessing_fcn is None:
        preprocessing_fcn = get_preprocessing(backbone)

    if input_size is not None:
        image = resize(image, input_size)
    image = image / 255
    image = preprocessing_fcn(image)
    return np.expand_dims(image, 0)


def predict(model, image, out_path=None, backbone='resnet34',
            preprocessing_fcn=None, input_size=None):

    def quantize(mask):
        mask = mask * 255
        return mask.squeeze().astype('uint8')

    x = preprocess(image=image,
                   input_size=input_size,
                   backbone=backbone,
                   preprocessing_fcn=preprocessing_fcn)
    y = model.predict(x)

    if out_path is not None:
        imsave(out_path, quantize(y))
    return y


def predict_all(model, data_dir, out_dir='results', backbone='resnet34',
                input_size=None, preprocessing_fcn=None):
    if preprocessing_fcn is None:
        preprocessing_fcn = get_preprocessing(backbone)

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    images = []
    for directory, _, files in os.walk(data_dir):
        for fn in files:
            if is_image(fn):
                images.append(os.path.join(directory, fn))

    for fn_full in images:
        name = os.path.splitext(os.path.basename(fn_full))[0]
        predict(model=model,
                image=fn_full,
                out_path=os.path.join(out_dir, name + '.png'),
                preprocessing_fcn=preprocessing_fcn,
                input_size=input_size)


def train(data_dir, model=None, backbone='resnet34', encoder_weights='imagenet',
          batch_size=2, all_layer_epochs=100, decode_only_epochs=2,
          logdir='logs', run_name='fish', verbose=2,
          patience=10, checkpoint_path='model_fish.h5', optimizer='adam',
          input_size=None, keras_augmentations=None,
          preprocessing_function_x=None, preprocessing_function_y=None,
          debug_training_data=False, debug_validation_data=False,
          preload=False, cached_preloading=False,
          visual_validation_samples=None, datumbox_mode=False,
          random_crops=False, learning_rate=None, n_gpus=1):
    # get data generators
    (training_generator, validation_generator,
     training_steps_per_epoch, validation_steps_per_epoch) = \
        get_data_generators(data_dir=data_dir,
                            backbone=backbone,
                            batch_size=batch_size,
                            input_size=input_size,
                            keras_augmentations=keras_augmentations,
                            preprocessing_function_x=preprocessing_function_x,
                            preprocessing_function_y=preprocessing_function_y,
                            preload=preload,
                            cached_preloading=cached_preloading,
                            random_crops=random_crops)

    # show images as they're input into model
    if debug_training_data:
        show_generated_pairs(training_generator)
        return
    if debug_validation_data:
        show_generated_pairs(validation_generator)
        return

    # initialize model
    if datumbox_mode and model is None:
        print('\n\nRunning in datumbox mode...\n\n')
        try:
            from datumbox_unet import DatumboxUnet
        except:
            from .datumbox_unet import DatumboxUnet
        model = DatumboxUnet(backbone_name=backbone,
                             encoder_weights=encoder_weights,
                             encoder_freeze=True)
    elif model is None:
        model = Unet(backbone_name=backbone,
                     encoder_weights=encoder_weights,
                     encoder_freeze=True)
    elif isinstance(model, str):
        model = load_model(model)

    if learning_rate is not None:
        if optimizer == 'adam':
            optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999,
                             epsilon=None, decay=0.0, amsgrad=False)
        else:
            raise NotImplementedError(
                'Adjustable learning rate not implemented for %s.' % optimizer)

    if n_gpus > 1:
        from keras.utils import multi_gpu_model
        model = multi_gpu_model(model, gpus=n_gpus)

    model.compile(optimizer, loss=bce_jaccard_loss, metrics=[iou_score])
    # model.compile(optimizer, 'binary_crossentropy', ['binary_accuracy'])

    # get callbacks
    callbacks = get_callbacks(
        checkpoint_path=checkpoint_path,
        verbose=verbose,
        batch_size=batch_size,
        patience=patience,
        logdir=logdir,
        run_name=run_name,
        visual_validation_samples=visual_validation_samples,
        steps_per_report=training_steps_per_epoch,
        steps_per_epoch=training_steps_per_epoch,
        epochs=all_layer_epochs,
        input_size=input_size,
        predict_fcn=predict)

    # train for `decoder_only_epochs` epochs with encoder frozen
    if decode_only_epochs:
        print('\n\nTraining decoder (only) for %s epochs...\n'
              '' % decode_only_epochs)
        model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            validation_steps=int(validation_steps_per_epoch),
                            steps_per_epoch=int(training_steps_per_epoch),
                            epochs=decode_only_epochs,
                            callbacks=list(callbacks.values()))

    # train all layers
    if all_layer_epochs:

        # refresh early stopping callback
        callbacks['EarlyStopping'] = \
            get_callbacks(patience=patience, verbose=verbose)['EarlyStopping']

        print('\n\nTraining all layers for %s epochs...\n' % all_layer_epochs)
        set_trainable(model)  # set all layers trainable and recompile model
        model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            validation_steps=int(validation_steps_per_epoch),
                            steps_per_epoch=int(training_steps_per_epoch),
                            epochs=all_layer_epochs,
                            callbacks=list(callbacks.values()),
                            initial_epoch=decode_only_epochs)

    # evaluate on training data
    print('\n\nTraining Scores\n' + '-'*14)
    results = model.evaluate_generator(generator=training_generator,
                                       steps=training_steps_per_epoch,
                                       max_queue_size=10,
                                       workers=1,
                                       use_multiprocessing=False,
                                       verbose=0)
    for name, value in zip(model.metrics_names, list(results)):
        print(name + ':', value)

    # evaluate on training data
    print('\n\nValidation Scores\n' + '-'*16)
    results = model.evaluate_generator(generator=training_generator,
                                       steps=validation_steps_per_epoch,
                                       max_queue_size=10,
                                       workers=1,
                                       use_multiprocessing=False,
                                       verbose=0)
    for name, value in zip(model.metrics_names, list(results)):
        print(name + ':', value)
    return model


def evaluate(data_dir, model=None, backbone='resnet34', batch_size=2,
             input_size=None, n_gpus=1,
             preprocessing_function_x=None, preprocessing_function_y=None):

    # initialize model
    if isinstance(model, str):
        model = load_model(model)

    if n_gpus > 1:
        from keras.utils import multi_gpu_model
        model = multi_gpu_model(model, gpus=n_gpus)

    # get data generators
    (training_generator, validation_generator,
     training_steps_per_epoch, validation_steps_per_epoch) = \
        get_data_generators(data_dir=data_dir,
                            backbone=backbone,
                            batch_size=batch_size,
                            input_size=input_size,
                            keras_augmentations=None,
                            preprocessing_function_x=preprocessing_function_x,
                            preprocessing_function_y=preprocessing_function_y,
                            preload=False,
                            cached_preloading=False,
                            random_crops=False)

    # evaluate on training data
    print('\n\nTraining Scores\n' + '-'*14)
    results = model.evaluate_generator(generator=training_generator,
                                       steps=None,
                                       max_queue_size=10,
                                       workers=1,
                                       use_multiprocessing=False,
                                       verbose=0)
    for name, value in zip(model.metrics_names, list(results)):
        print(name + ':', value)

    # evaluate on training data
    print('\n\nValidation Scores\n' + '-'*16)
    results = model.evaluate_generator(generator=training_generator,
                                       steps=None,
                                       max_queue_size=10,
                                       workers=1,
                                       use_multiprocessing=False,
                                       verbose=0)
    for name, value in zip(model.metrics_names, list(results)):
        print(name + ':', value)


DEFAULT_KERAS_AUGMENTATIONS = dict(rotation_range=20,  # used to be 0.2
                                   width_shift_range=0.05,
                                   height_shift_range=0.05,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='constant')


if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("command",
                      help="'train' or 'predict' (file or directory)")
    args.add_argument("input",
                      help="A file or directory.\nIf `command` is "
                           "'train', then this must be a directory "
                           "containing 'images' and 'segmentations' "
                           "subdirectories.\nIf `command` is 'predict', "
                           "then this can be a file or directory.")
    args.add_argument('-o', "--output",
                      help="Path to use for output file/directory.")
    args.add_argument("--epochs", default=100, type=int,
                      help="Number of training epochs.")
    args.add_argument("--decoder_only_epochs", default=2, type=int,
                      help="Number of epochs freeze the encoder for when "
                           "starting training.")
    args.add_argument("--batch_size", default=2, type=int,
                      help="Number of images per training batch.")
    args.add_argument('-l', "--logdir",
                      default=os.path.join(tempdir, 'unet-logs'),
                      help="Where to store logs.")
    args.add_argument('-k', "--checkpoint_path",
                      default=os.path.join(tempdir, 'unet-checkpoint.h5'),
                      help="Where to store logs.")
    args.add_argument('-n', "--run_name", default='unnamed_%s' % time(),
                      help="Name for this run/experiment.")
    args.add_argument("--verbosity", default=2, type=int,
                      help="Verbosity setting.")
    args.add_argument("--patience", default=10, type=int,
                      help="Patience for early stopping.")
    args.add_argument('-a', "--augment", default=False, action='store_true',
                      help="Invoke to use augmentation.")
    args.add_argument("--initial_model", default=None,
                      help="Keras model to start with.")
    args.add_argument("--backbone", default='resnet34',
                      help="Model (encoder) backbone.")
    args.add_argument("--input_size", nargs=2, default=(1024, 1024), type=int,
                      help="Input size (will resize if necessary).")
    args.add_argument("--show_training_data",
                      default=False, action='store_true',
                      help="Show training data.")
    args.add_argument("--show_validation_data",
                      default=False, action='store_true',
                      help="Show validation data.")
    args.add_argument('-V', "--visual_validation_samples",
                      default=None, nargs='+',
                      help="Image to use for visual validation.  Output "
                           "stored in <logdir>/visual/<run_name>")
    args.add_argument("--datumbox", default=False, action='store_true',
                      help="Invoke when using datumbox_keras.")
    args.add_argument('-r', "--learning_rate", default=None, type=float,
                      help="Learning rate for optimizer.")
    args.add_argument("--n_gpus", default=1, type=int,
                      help="How many GPUs are available for use.")
    args.add_argument(
        '-p', "--preload", default=False, action='store_true',
        help="If invoked, dataset will be preloaded.")
    args.add_argument(
        '-c', "--cached_preloading", default=False, action='store_true',
        help="Speed up preloading by looking for npy files stored in "
             "TMPDIR from previous runs.  This will speed things up "
             "significantly is only appropriate when you want to reuse "
             "the split dataset created in the last run.")
    args.add_argument(
        '-C', "--random_crops", default=0.0, type=float,
        help="Probability of sending random crop instead of whole image.")
    args = args.parse_args()

    args.input_size = tuple(args.input_size)
    w, h = args.input_size
    assert w//32 == w/32. and h//32 == h/32.

    if args.augment:
        keras_augmentations = DEFAULT_KERAS_AUGMENTATIONS
    else:
        keras_augmentations = None

    if not args.run_name:
        args.run_name = 'unnamed_%s' % time()

    if args.preload or args.cached_preloading:
        print("\n\n Preloading requires unsplit dataset functionality to "
              "be implemented.\n\n")
        raise NotImplementedError

    # record user arguments in log directory
    run_logs_dir = os.path.join(args.logdir, args.run_name)
    if not os.path.exists(run_logs_dir):
        os.makedirs(run_logs_dir)
    with open(os.path.join(run_logs_dir, 'args.txt'), 'w+') as f:
        s = '\n'.join("%s: %s" % (key, val) for key, val in vars(args).items())
        f.write(s)

    # go
    if args.command == 'train':
        m = train(data_dir=args.input,
                  model=args.initial_model,
                  backbone=args.backbone,
                  batch_size=args.batch_size,
                  all_layer_epochs=args.epochs - args.decoder_only_epochs,
                  decode_only_epochs=args.decoder_only_epochs,
                  logdir=args.logdir,
                  run_name=args.run_name,
                  verbose=args.verbosity,
                  patience=args.patience,
                  checkpoint_path=args.checkpoint_path,
                  input_size=args.input_size,
                  encoder_weights='imagenet',
                  optimizer='adam',
                  keras_augmentations=keras_augmentations,
                  debug_training_data=args.show_training_data,
                  debug_validation_data=args.show_validation_data,
                  preload=args.preload,
                  cached_preloading=args.cached_preloading,
                  visual_validation_samples=args.visual_validation_samples,
                  datumbox_mode=args.datumbox,
                  random_crops=args.random_crops,
                  learning_rate=args.learning_rate,
                  n_gpus=args.n_gpus)
    elif args.command == 'evaluate':
        m = load_model(args.checkpoint_path)
        evaluate(data_dir=args.input,
                 model=m,
                 backbone=args.backbone,
                 batch_size=args.batch_size,
                 input_size=args.input_size,
                 n_gpus=args.n_gpus)

    elif args.command == 'predict':
        m = load_model(args.checkpoint_path)
        if os.path.isdir(args.input):
            predict_all(model=m,
                        data_dir=args.input,
                        out_dir=args.output,
                        input_size=args.input_size,
                        backbone=args.backbone,
                        preprocessing_fcn=get_preprocessing(args.backbone))
        else:
            predict(model=m,
                    image=args.input,
                    out_path=args.output,
                    backbone=args.backbone,
                    preprocessing_fcn=get_preprocessing(args.backbone),
                    input_size=args.input_size)
    else:
        raise ValueError('`command` = "%s" not understood.' % args.command)
