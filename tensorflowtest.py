import tensorflow as tf
import numpy
print(numpy.version.version)
print(tf.config.list_physical_devices('GPU'))
tf.test.is_gpu_available(cuda_only=True)

import tfimm
# from models_vit import *
# # call the model
# keras_model = tfimm.create_model( # apply global pooling without class token
#     "vit_large_patch16_224_mae",
#     nb_classes = 5
#     )

# layers = keras_model.layers

# print(keras_model.summary())
# print(tfimm.list_models())