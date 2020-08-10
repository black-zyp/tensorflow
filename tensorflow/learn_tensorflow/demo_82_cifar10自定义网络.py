import tensorflow as tf
from tensorflow.keras import Sequential, layers, optimizers, metrics, datasets
from tensorflow import keras


def process(x, y):
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(x, y), (x_test, y_test) = datasets.cifar10.load_data()
y = tf.squeeze(y)
y_test = tf.squeeze(y_test)
y = tf.one_hot(y, depth=10)
y_test = tf.one_hot(y_test, depth=10)

batch_size = 128
db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(process).shuffle(10000).batch(batch_size)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(process).batch(batch_size)


class MyDense(layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(MyDense, self).__init__()
        self.kernel = self.add_variable('w', [input_dim, output_dim])
        # self.bais = self.add_variable('b', output_dim)

    def call(self, inputs, training=None):
        x = inputs @ self.kernel
        return x


class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.net1 = MyDense(32 * 32 * 3, 512)
        self.net2 = MyDense(512, 256)
        self.net3 = MyDense(256, 128)
        self.net4 = MyDense(128, 64)
        self.net5 = MyDense(64, 32)
        self.net6 = MyDense(32, 10)

    def call(self, inputs, training=None):
        x = tf.reshape(inputs, [-1, 32 * 32 * 3])
        x = self.net1(x)
        x = tf.nn.relu(x)
        x = self.net2(x)
        x = tf.nn.relu(x)
        x = self.net3(x)
        x = tf.nn.relu(x)
        x = self.net4(x)
        x = tf.nn.relu(x)
        x = self.net5(x)
        x = tf.nn.relu(x)
        x = self.net6(x)

        return x


net_work = MyModel()

net_work.compile(
    optimizer=optimizers.Adam(1e-3),
    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

net_work.fit(db, epochs=20, validation_data=db_test, validation_freq=2)

net_work.evaluate(db_test)

net_work.save_weights('./save_model2/model')

del net_work

print("already save model!")

net_work = MyModel()

net_work.compile(
    optimizer=optimizers.Adam(1e-3),
    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

net_work.load_weights('./save_model2/model')

print('loading save model!')

net_work.evaluate(db_test)
