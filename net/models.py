import tensorflow as tf
from tensorflow.python.keras.layers import Conv3D, Conv3DTranspose, LeakyReLU, BatchNormalization, Activation

class Encoder_3DCNN():
    def __init__(self, dtype):
        super().__init__()
        self.conv1 = Conv3D(32, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding='SAME', dtype=dtype, activation=None, name='conv1')
        self.bn1 = BatchNormalization(momentum=0.999, epsilon=1e-5, dtype=dtype, name='bn1')
        self.lrelu1 = LeakyReLU(alpha=0.02, name='lrelu1')

        self.conv2 = Conv3D(64, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding='SAME', dtype=dtype, activation=None, name='conv2')
        self.bn2 = BatchNormalization(momentum=0.999, epsilon=1e-5, dtype=dtype, name='bn2')
        self.lrelu2 = LeakyReLU(alpha=0.02, name='lrelu2')

        self.conv3 = Conv3D(128, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding='SAME', dtype=dtype, activation=None, name='conv3')
        self.bn3 = BatchNormalization(momentum=0.999, epsilon=1e-5, dtype=dtype, name='bn3')
        self.lrelu3 = LeakyReLU(alpha=0.02, name='lrelu3') 

        self.conv4 = Conv3D(256, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding='SAME', dtype=dtype, activation=None, name='conv4')
        self.bn4 = BatchNormalization(momentum=0.999, epsilon=1e-5, dtype=dtype, name='bn4')
        self.lrelu4 = LeakyReLU(alpha=0.02, name='lrelu4') 

        self.conv5 = Conv3D(128, kernel_size=(4, 4, 4), padding='VALID', dtype=dtype, activation=None, name='conv5')

        self.out = Activation(activation=tf.keras.activations.sigmoid, name='sigmoid')

    def build(self, inputs):
        net = self.conv1(inputs)
        net = self.bn1(net)
        net = self.lrelu1(net)

        net = self.conv2(net)
        net = self.bn2(net)
        net = self.lrelu2(net)

        net = self.conv3(net)
        net = self.bn3(net)
        net = self.lrelu3(net)

        net = self.conv4(net)
        net = self.bn4(net)
        net = self.lrelu4(net)

        net = self.conv5(net)
        
        outputs = self.out(net)

        return tf.keras.Model(inputs=inputs, outputs=outputs)

class Decoder_3DCNN():
    def __init__(self, dtype):
        super().__init__()
        self.deconv1 = Conv3DTranspose(256, kernel_size=(4, 4, 4), padding='VALID', dtype=dtype, activation=None, name='deconv1')
        self.bn1 = BatchNormalization(momentum=0.999, epsilon=1e-5, dtype=dtype, name='bn1')
        self.lrelu1 = LeakyReLU(alpha=0.02, name='lrelu1')

        self.deconv2 = Conv3DTranspose(128, kernel_size=(4, 4, 4), strides=(2 , 2, 2), padding='SAME', dtype=dtype, activation=None, name='deconv2')
        self.bn2 = BatchNormalization(momentum=0.999, epsilon=1e-5, dtype=dtype, name='bn2')
        self.lrelu2 = LeakyReLU(alpha=0.02, name='lrelu2')

        self.deconv3 = Conv3DTranspose(64, kernel_size=(4, 4, 4), strides=(2 , 2, 2), padding='SAME', dtype=dtype, activation=None, name='deconv3')
        self.bn3 = BatchNormalization(momentum=0.999, epsilon=1e-5, dtype=dtype, name='bn3')
        self.lrelu3 = LeakyReLU(alpha=0.02, name='lrelu3')

        self.deconv4 = Conv3DTranspose(32, kernel_size=(4, 4, 4), strides=(2 , 2, 2), padding='SAME', dtype=dtype, activation=None, name='deconv4')
        self.bn4 = BatchNormalization(momentum=0.999, epsilon=1e-5, dtype=dtype, name='bn4')
        self.lrelu4 = LeakyReLU(alpha=0.02, name='lrelu4')
    
        self.deconv5 = Conv3DTranspose(1, kernel_size=(4, 4, 4), strides=(2 , 2, 2), padding='SAME', dtype=dtype, activation=None, name='deconv5')

        self.out = Activation(activation=tf.keras.activations.sigmoid, name='sigmoid')

    def build(self, inputs):
        net = self.deconv1(inputs)
        net = self.bn1(net)
        net = self.lrelu1(net)

        net = self.deconv2(net)
        net = self.bn2(net)
        net = self.lrelu2(net)

        net = self.deconv3(net)
        net = self.bn3(net)
        net = self.lrelu3(net)

        net = self.deconv4(net)
        net = self.bn4(net)
        net = self.lrelu4(net)

        net = self.deconv5(net)

        outputs = self.out(net)

        return tf.keras.Model(inputs=inputs, outputs=outputs)