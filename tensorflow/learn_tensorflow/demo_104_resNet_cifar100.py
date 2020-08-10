import tensorflow as tf
from tensorflow.keras import Sequential, layers, optimizers, datasets, metrics
from tensorflow import keras


class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()

        # 维度可能需要-也可能不需要变换,取决于 stride
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        # BN 层是一种防止过拟合的正则化的手段
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        # 维度不需要变换
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        # 这一步就是为了 让他们维度相同
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x: x

    def call(self, inputs, **kwargs):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(inputs)

        out_put = layers.add([out, identity])
        out_put = tf.nn.relu(out_put)

        return out_put


class ResNet(keras.Model):
    def __init__(self, layer_dims, num_classer=100): #[2,2,2,2 ]
        super(ResNet, self).__init__()
        self.stem = Sequential([
            layers.Conv2D(64, (3, 3), strides=(1, 1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
        ])

        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classer)

    def call(self, inputs, training=None):
        x = self.stem(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(filter_num, stride))

        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks


def resnet18():
    return ResNet([2, 2, 2, 2])


def resnet34():
    return ResNet([3, 4, 6, 8])
