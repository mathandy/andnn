from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
import os

from iotools import get_data_generators


DATA_DIR = os.path.expanduser('~/Dropbox/hand-segmented-fish-new-old-split-hiRes')
BACKBONE = 'resnet34'
preprocess_input = get_preprocessing(BACKBONE)

# load your data
# x_train, y_train, x_val, y_val = load_data(...)
(training_generator, validation_generator,
            training_steps_per_epoch, validation_steps_per_epoch) = \
    get_data_generators(DATA_DIR,
                        backbone=BACKBONE,
                        batch_size=2,
                        # input_size=(512, 768),
                        input_size=(1536, 768),
                        # input_size=(3300, 1452),
                        keras_augmentations=None,
                        preprocessing_function_x=None,
                        preprocessing_function_y=None,
                        preload=False,
                        cached_preloading=False,
                        presplit=True,
                        random_crops=False)

# preprocess input
# x_train = preprocess_input(x_train)
# x_val = preprocess_input(x_val)

# define model
model = Unet(BACKBONE, encoder_weights='imagenet')
model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])

model.fit_generator(training_generator,
                    steps_per_epoch=training_steps_per_epoch,
                    epochs=1,
                    verbose=1,
                    callbacks=None,
                    validation_data=validation_generator,
                    validation_steps=validation_steps_per_epoch,
                    class_weight=None,
                    max_queue_size=10,
                    workers=1,
                    use_multiprocessing=False,
                    shuffle=True,
                    initial_epoch=0)

# fit model
# model.fit(
#     x=x_train,
#     y=y_train,
#     batch_size=16,
#     epochs=100,
#     validation_data=(x_val, y_val),
# )
