import tensorflow as tf
from tensorflow.keras import Sequential, layers, optimizers, datasets, metrics
from tensorflow import keras


class restnet(layers.Layer):
    def __init__(self, file_num, stride=1):
        super(restnet, self).__init__()
        self.conv1 = layers.Conv2D(file_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(file_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(file_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(inputs)

        out_put = layers.add([out, identity])
        out_put = tf.nn.relu(out_put)
        return out_put


class baserestnet(keras.Model):
    def __init__(self, input_dims, num_classes=100):
        super(baserestnet, self).__init__()
        self.stem = Sequential([layers.Conv2D(64, (3, 3), strides=(1, 1)),
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                layers.MaxPool2D((2, 2), strides=(1, 1), padding='same')
                                ])

        self.layer1 = self.build_big(64, input_dims[0])
        self.layer2 = self.build_big(128, input_dims[1], stride=2)
        self.layer3 = self.build_big(256, input_dims[2], stride=2)
        self.layer4 = self.build_big(512, input_dims[3], stride=2)

        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=None):
        x = self.stem(inputs)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x

    def build_big(self, file_num, block, stride=1):
        rest_block = Sequential()
        rest_block.add(restnet(file_num, stride))

        for _ in range(1, block):
            rest_block.add(restnet(file_num, stride=1))

        return rest_block


def resnet18():
    return baserestnet([2, 2, 2, 2])


def resnet34():
    return baserestnet([3, 4, 6, 3])
