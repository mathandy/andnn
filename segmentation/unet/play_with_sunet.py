from unet_with_channelwise_softmax import Unet
from keras.layers import Lambda
from keras import backend as K

N_CLASSES = 10


def channelwise_softmax(x):
    # tf.exp(x)/tf.reduce_sum(tf.exp(x), axis=(0, 1))[None, None, :]
    return K.exp(x) / K.sum(K.exp(x), axis=(0, 1))[None, None, :]


model = Unet(backbone_name='resnet34',
             encoder_weights='imagenet',
             encoder_freeze=True,
             activation=Lambda(channelwise_softmax),
             classes=N_CLASSES)