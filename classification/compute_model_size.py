"""
Credit: https://stackoverflow.com/questions/43137288
"""

def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes


if __name__ == '__main__':
    from segmentation_models import Unet

    INPUT_SHAPE = (1536, 768, 3)
    BATCH_SIZE = 3
    m = Unet(backbone_name='resnet34',
             encoder_weights='imagenet',
             input_shape=INPUT_SHAPE)
    gb = get_model_memory_usage(BATCH_SIZE, m)
    print(gb, 'GiB')
