import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, ReLU, UpSampling2D, LeakyReLU


class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(0.0, 0.02),
            trainable=True,
        )

        self.offset = self.add_weight(
            name="offset",
            shape=input_shape[-1:],
            initializer="zeros",
            trainable=True,
        )

    def call(self, x, **kwargs):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset


class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=1, **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.padding = padding

    def compute_output_shape(self, s):
        return s[0], s[1] + 2 * self.padding, s[2] + 2 * self.padding, s[3]

    def call(self, x, **kwargs):
        return tf.pad(
            x,
            [
                [0, 0],
                [self.padding, self.padding],
                [self.padding, self.padding],
                [0, 0],
            ],
            "REFLECT",
        )


class ResidualBlock(tf.keras.Model):
    def __init__(self, channels, strides=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, kernel_size=3, strides=strides)
        self.in1 = InstanceNormalization()
        self.conv2 = ConvLayer(channels, kernel_size=3, strides=strides)
        self.in2 = InstanceNormalization()

    def call(self, inputs, training=None, **kwargs):
        residual = inputs

        x = self.in1(self.conv1(inputs))
        x = tf.nn.relu(x)

        x = self.in2(self.conv2(x))
        x = x + residual
        return x


class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, channels, kernel_size=3, strides=1):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = ReflectionPadding2D(reflection_padding)
        self.conv2d = Conv2D(channels, kernel_size, strides=strides)

    def call(self, x, **kwargs):
        x = self.reflection_pad(x)
        x = self.conv2d(x)
        return x


class UpsampleConvLayer(tf.keras.layers.Layer):
    def __init__(self, channels, kernel_size=3, strides=1, upsample=2):
        super(UpsampleConvLayer, self).__init__()
        self.up2d = UpSampling2D(size=upsample)
        self.conv2d = Conv2D(channels, kernel_size, strides=strides, padding="same")

    def call(self, x, **kwargs):
        x = self.up2d(x)
        x = self.conv2d(x)
        return x


class GeneratorNet(tf.keras.Model):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        self.conv1 = ConvLayer(64, kernel_size=7, strides=1)
        self.in1 = InstanceNormalization()
        self.conv2 = Conv2D(64, kernel_size=3, strides=2, padding="same")
        self.in2 = InstanceNormalization()
        self.conv3 = Conv2D(128, kernel_size=3, strides=2, padding="same")
        self.in3 = InstanceNormalization()

        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.res6 = ResidualBlock(128)

        self.deconv1 = UpsampleConvLayer(
            64, kernel_size=3, strides=1, upsample=2
        )
        self.in4 = InstanceNormalization()
        self.deconv2 = UpsampleConvLayer(
            32, kernel_size=3, strides=1, upsample=2
        )
        self.in5 = InstanceNormalization()
        self.deconv3 = ConvLayer(3, kernel_size=7, strides=1)

        self.relu = ReLU()

    def call(self, x, **kwargs):
        x = self.relu(self.in1(self.conv1(x)))
        x = self.relu(self.in2(self.conv2(x)))
        x = self.relu(self.in3(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.relu(self.in4(self.deconv1(x)))
        x = self.relu(self.in5(self.deconv2(x)))
        x = self.deconv3(x)
        x = tf.nn.tanh(x)
        return x


class DiscriminatorNet(tf.keras.Model):
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        self.conv1 = Conv2D(64, kernel_size=4, strides=2, padding="same", use_bias=False)

        self.conv2 = Conv2D(128, kernel_size=4, strides=2, padding="same", use_bias=False)
        self.in2 = InstanceNormalization()
        self.conv3 = Conv2D(256, kernel_size=4, strides=2, padding="same", use_bias=False)
        self.in3 = InstanceNormalization()
        self.conv4 = Conv2D(512, kernel_size=4, strides=2, padding="same", use_bias=False)
        self.in4 = InstanceNormalization()

        self.conv5 = Conv2D(1, kernel_size=4, strides=1, padding="same")

        self.lrelu = LeakyReLU()

    def call(self, x, **kwargs):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.in2(self.conv2(x)))
        x = self.lrelu(self.in3(self.conv3(x)))
        x = self.lrelu(self.in4(self.conv4(x)))
        x = self.conv5(x)
        return x
