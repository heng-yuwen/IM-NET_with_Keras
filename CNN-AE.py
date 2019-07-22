import tensorflow as tf
from tensorflow.python.keras import backend as K
from net.models import Encoder_3DCNN, Decoder_3DCNN

tf.enable_eager_execution()

encoder_3dcnn = Encoder_3DCNN(tf.float32)
encoder_3dcnn = encoder_3dcnn.build(inputs=tf.keras.layers.Input(shape=(64, 64, 64, 1)))
encoder_3dcnn.summary()

decoder_3dcnn = Decoder_3DCNN(tf.float32)
decoder_3dcnn = decoder_3dcnn.build(inputs=tf.keras.layers.Input(shape=(1, 1, 1, 128)))
decoder_3dcnn.summary()

K.set_learning_phase(0) # inference mode