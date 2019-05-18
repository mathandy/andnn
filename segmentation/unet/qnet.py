"""A U-Net based single-stage polygonal detection model.

Author:
    Andy Port

Credit:
    Based on code from `segmentation_models` and, in-part, inspired by
    the Retina U-Net (arXiv:1811.08661) model and the medical detection
    toolkit, https://github.com/pfjaeger/medicaldetectiontoolkit
"""


from keras.models import Model
from keras.layers import Conv2D, Activation
from segmentation_models.utils import (
    get_layer_number, to_tuple, freeze_model, legacy_support)
from segmentation_models.backbones import get_backbone, get_feature_layers
from segmentation_models.unet.blocks import Transpose2D_block, Upsample2D_block
import keras.backend as K


class PPNHead:
    def __init__(self, degree, proposals_per_anchor, n_classes, n_features=256,
                 propose_homographies=False):
        self.degree = degree
        self.proposals_per_anchor = proposals_per_anchor
        self.n_classes = n_classes
        self.n_features = n_features
        self.propose_homographies = propose_homographies

    def __call__(self, feature_map, level_id):
        """Polygon Proposal Network (PPN)

        Args:
            feature_map (tensor): feature map of shape
                [batch_size, height, width, channels]
            degree (int): degree of polygons to be proposed
            proposals_per_anchor (int): number of proposals per anchor
            n_classes (int): number of object classes
            n_features (int): number of output channels for hidden conv layers
            propose_homographies (bool): if true, output will include
            homography proposal

        Returns:
            (tensor): 2-tensor of vertex coordinates of shape
                [batch_size, height, width, proposals_per_anchor, 2 * degree]
            (tensor): 2-tensor of classification probabilities of shape
                [batch_size, height, width, proposals_per_anchor, n_classes]
            (tensor or None): None if `propose_homographies` is false,
                otherwise a 2-tensor of shape
                [batch_size, height, width, proposals_per_anchor, 8]
        """
        batch_size, height, width, _ = K.shape(feature_map)
        n_poly = 2 * self.degree * self.proposals_per_anchor
        n_cl = self.n_classes * self.proposals_per_anchor
        n_h = 8 * self.proposals_per_anchor
        nf = self.n_features

        p = Conv2D(nf, (3, 3), padding='same', activation='relu')(feature_map)
        p = Conv2D(nf, (3, 3), padding='same', activation='relu')(p)
        p = Conv2D(nf, (3, 3), padding='same', activation='relu')(p)
        p = Conv2D(nf, (3, 3), padding='same', activation='relu')(p)
        p = Conv2D(n_poly, (3, 3), padding='same', activation='sigmoid', name='poly_output')(p)
        K.reshape(p, (batch_size, height, width, self.proposals_per_anchor, 2 * self.degree))

        cl = Conv2D(nf, (3, 3), padding='same', activation='relu')(feature_map)
        cl = Conv2D(nf, (3, 3), padding='same', activation='relu')(cl)
        cl = Conv2D(nf, (3, 3), padding='same', activation='relu')(cl)
        cl = Conv2D(nf, (3, 3), padding='same', activation='relu')(cl)
        cl = Conv2D(n_cl, (3, 3), padding='same', activation='sigmoid', name='class_output')(cl)
        K.reshape(cl, (batch_size, height, width, self.proposals_per_anchor, self.n_classes))

        h = None
        if self.propose_homographies:
            h = Conv2D(nf, (3, 3), padding='same', activation='relu', name='hom_conv_1')(feature_map)
            h = Conv2D(nf, (3, 3), padding='same', activation='relu', name='hom_conv_2')(h)
            h = Conv2D(nf, (3, 3), padding='same', activation='relu', name='hom_conv_3')(h)
            h = Conv2D(nf, (3, 3), padding='same', activation='relu', name='hom_conv_4')(h)
            h = Conv2D(n_h, (3, 3), padding='same', activation='sigmoid', name='homography_output')(h)
            K.reshape(h, (batch_size, height, width, self.proposals_per_anchor, 8))
        return p, cl, h


def build_qnet(backbone, classes, skip_connection_layers,
               decoder_filters=(256,128,64,32,16),
               upsample_rates=(2,2,2,2,2),
               n_upsample_blocks=5,
               block_type='upsampling',
               activation='sigmoid',
               use_batchnorm=True,
               propose_homographies=True,
               degree=None,
               proposals_per_anchor=None,
               n_classes=None):

    assert (degree is not None and
            proposals_per_anchor is not None and
            n_classes is not None)

    input = backbone.input
    x = backbone.output

    if block_type == 'transpose':
        up_block = Transpose2D_block
    else:
        up_block = Upsample2D_block

    # convert layer names to indices
    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                               for l in skip_connection_layers])

    decoder_feature_maps = []
    for i in range(n_upsample_blocks):

        # check if there is a skip connection
        skip_connection = None
        if i < len(skip_connection_idx):
            skip_connection = backbone.layers[skip_connection_idx[i]].output

        # collect decoder feature maps for PPN
        decoder_feature_maps.append(x)

        # upsample
        upsample_rate = to_tuple(upsample_rates[i])

        x = up_block(decoder_filters[i], i, upsample_rate=upsample_rate,
                     skip=skip_connection, use_batchnorm=use_batchnorm)(x)

    masks = Conv2D(classes, (3, 3), padding='same', activation=activation,
                   name='mask_output')(x)
    # if isinstance(activation, str):
    #     masks = Activation(activation, name=activation)(x)
    # else:
    #     masks = activation(x)

    # get detections and classifications
    polygons = []
    classifications = []
    homographies = []
    ppn_head = PPNHead(degree=degree,
                       proposals_per_anchor=proposals_per_anchor,
                       n_classes=n_classes,
                       n_features=256,
                       propose_homographies=propose_homographies)
    for i, feature_map in enumerate(decoder_feature_maps):
        q, c, h = ppn_head(feature_map, i)
        polygons.append(q)
        classifications.append(c)
        homographies.append(h)
    polygons = K.concatenate(polygons, -1)
    classifications = K.concatenate(classifications, -1)
    if propose_homographies:
        homographies = K.concatenate(homographies, -1)

    if propose_homographies:
        return Model(inputs=input,
                     outputs=[masks, polygons, classifications, homographies])
    else:
        return Model(inputs=input,
                     outputs=[masks, polygons, classifications])


old_args_map = {
    'freeze_encoder': 'encoder_freeze',
    'skip_connections': 'encoder_features',
    'upsample_rates': None,  # removed
    'input_tensor': None,  # removed
}


@legacy_support(old_args_map)
def Qnet(backbone_name='vgg16',
         input_shape=(None, None, 3),
         classes=1,
         activation='sigmoid',
         encoder_weights='imagenet',
         encoder_freeze=False,
         encoder_features='default',
         decoder_block_type='upsampling',
         decoder_filters=(256, 128, 64, 32, 16),
         decoder_use_batchnorm=True,
         **kwargs):
    """ Unet_ is a fully convolution neural network for image semantic segmentation

        Args:
            backbone_name: name of classification model (without last dense layers) used as feature
                extractor to build segmentation model.
            input_shape: shape of input data/image ``(H, W, C)``, in general
                case you do not need to set ``H`` and ``W`` shapes, just pass ``(None, None, C)`` to make your model be
                able to process images af any size, but ``H`` and ``W`` of input images should be divisible by factor ``32``.
            classes: a number of classes for output (output shape - ``(h, w, classes)``).
            activation: name of one of ``keras.activations`` for last model layer
                (e.g. ``sigmoid``, ``softmax``, ``linear``).
            encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
            encoder_freeze: if ``True`` set all layers of encoder (backbone model) as non-trainable.
            encoder_features: a list of layer numbers or names starting from top of the model.
                Each of these layers will be concatenated with corresponding decoder block. If ``default`` is used
                layer names are taken from ``DEFAULT_SKIP_CONNECTIONS``.
            decoder_block_type: one of blocks with following layers structure:

                - `upsampling`:  ``Upsampling2D`` -> ``Conv2D`` -> ``Conv2D``
                - `transpose`:   ``Transpose2D`` -> ``Conv2D``

            decoder_filters: list of numbers of ``Conv2D`` layer filters in decoder blocks
            decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
                is used.

        Returns:
            ``keras.models.Model``: **Unet**

        .. _Unet:
            https://arxiv.org/pdf/1505.04597

    """

    backbone = get_backbone(backbone_name,
                            input_shape=input_shape,
                            input_tensor=None,
                            weights=encoder_weights,
                            include_top=False)

    if encoder_features == 'default':
        encoder_features = get_feature_layers(backbone_name, n=4)

    model = build_qnet(backbone,
                       classes,
                       encoder_features,
                       decoder_filters=decoder_filters,
                       block_type=decoder_block_type,
                       activation=activation,
                       n_upsample_blocks=len(decoder_filters),
                       upsample_rates=(2, 2, 2, 2, 2),
                       use_batchnorm=decoder_use_batchnorm)

    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone)

    model.name = 'u-{}'.format(backbone_name)

    return model



