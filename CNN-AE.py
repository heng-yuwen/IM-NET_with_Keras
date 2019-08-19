import tensorflow as tf
import numpy as np
import time
from scipy.io import loadmat
from tensorflow.python.keras import backend as K
from net.models import Encoder_3DCNN, Decoder_3DCNN

# tf.enable_eager_execution()

# encoder_3dcnn = Encoder_3DCNN(tf.float32)
# encoder_3dcnn = encoder_3dcnn.build(inputs=tf.keras.layers.Input(shape=(64, 64, 64, 1)))
# encoder_3dcnn.summary()

# decoder_3dcnn = Decoder_3DCNN(tf.float32)
# decoder_3dcnn = decoder_3dcnn.build(inputs=tf.keras.layers.Input(shape=(1, 1, 1, 128)))
# decoder_3dcnn.summary()

# K.set_learning_phase(0) # inference mode

# Explore the HSP file
voxel_model_mat = loadmat("/Users/hengyuwen/IM-NET_with_Keras/data/shapenet/modelBlockedVoxels256/02828884/1a40eaf5919b1b3f3eaa2b95b99dae6.mat")
print(type(voxel_model_mat))
print(voxel_model_mat.keys())