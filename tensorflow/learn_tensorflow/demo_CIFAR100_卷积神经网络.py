import tensorflow as tf
from tensorflow.keras import datasets, Sequential, optimizers, metrics, layers
from tensorflow import keras
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'


devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)

def process(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int64)
    return x, y


# Load datasets
(x, y), (x_test, y_test) = datasets.cifar100.load_data()
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)
print(x.shape)
db_train = tf.data.Dataset.from_tensor_slices((x, y))
db_train = db_train.shuffle(1000).map(process).batch(64)

db_test = tf.data.Dataset.from_tensor_slices((x, y))
db_test = db_test.map(process).batch(64)

sample = next(iter(db_train))
print(sample[0].shape, sample[1].shape)

# build network
conv_layers = [
    # 卷积层 取的是局部特征, 通过局部特征来表示
    layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    # 池化层 保留主要的特征同时减少参数(降维，效果类似PCA)和计算量，防止过拟合，提高模型泛化能力
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')
]

# sample = next(iter(db_train))
# print(sample[0].shape, sample[1].shape)


def main():
    # [b,32,32,3] ==> [b,1,1,512]
    conv_net = Sequential(conv_layers)
    conv_net.build(input_shape=[None, 32, 32, 3])
    # x = tf.random.normal([4,32,32,3])
    # out = conv_net(x)
    # print(out.shape)
    fc_net = Sequential([
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(100, activation=None)
    ])
    # conv_net.build(input_shape=[None, 32, 32, 3])
    fc_net.build(input_shape=[None, 512])

    optimizer = optimizers.Adam(lr=1e-4)

    variables = conv_net.trainable_variables + fc_net.trainable_variables

    acc_meter = metrics.Accuracy()
    loss_meter = metrics.Mean()

    for epoch in range(50):
        for step, (x, y) in enumerate(db_train):
            with tf.GradientTape() as tape:
                # [b,32,32,3] ==> [b,1,1,512]
                # print(x.shape)
                out = conv_net(x)
                print(out.shape)
                out = tf.reshape(out, [-1, 512])
                # [b,512] ==> [b,100]
                logits = fc_net(out)
                y_one_hot = tf.one_hot(y, depth=100)
                # print(y_one_hot.shape)
                # print(logits.shape)
                loss = tf.losses.categorical_crossentropy(y_one_hot, logits, from_logits=True)
                loss_meter.update_state(loss)

            grads = tape.gradient(loss, conv_net.trainable_variables + fc_net.trainable_variables)
            optimizer.apply_gradients(zip(grads, variables))

            if step % 100 == 0:
                print(epoch, step, 'loss:', loss_meter.result().numpy())
                loss_meter.reset_states()

        for step, (x, y) in enumerate(db_test):
            out = conv_net(x)
            # print(out.shape)
            out = tf.reshape(out, [-1, 512])
            # print(out.shape)
            logits = fc_net(out)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            acc_meter.update_state(y, pred)

        print(epoch, "acc result:", acc_meter.result().numpy())


if __name__ == '__main__':
    main()


# train

# test
