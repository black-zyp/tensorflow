import tensorflow as tf
from tensorflow.keras import datasets, Sequential, metrics, optimizers, layers
from tensorflow import keras


def process(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1, 28 * 28])
    y = tf.cast(y, dtype=tf.int64)
    y = tf.one_hot(y, depth=10)
    return x, y


(x, y), (x_test, y_test) = datasets.mnist.load_data()
# x_train, x_val = tf.split(x, num_or_size_splits=[50000, 10000])
# y_train, y_val = tf.split(y, num_or_size_splits=[50000, 10000])

batch_size = 128
db_train = tf.data.Dataset.from_tensor_slices((x, y))
db_train = db_train.map(process).shuffle(10000).batch(batch_size)

# db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
# db_val = db_val.map(process).shuffle(10000).batch(batch_size)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(process).batch(batch_size)


class MyDense(layers.Layer):
    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()
        self.kernel = self.add_variable('w', [inp_dim, outp_dim])
        self.bais = self.add_variable('b', [outp_dim])

    def call(self, inputs, **kwargs):
        out = inputs @ self.kernel + self.bais
        return out


class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.func_one = MyDense(784, 256)
        self.func_two = MyDense(256, 128)
        self.func_three = MyDense(128, 64)
        self.func_four = MyDense(64, 10)

    def call(self, inputs, training=None, mask=None):
        x = tf.reshape(inputs, [-1, 28 * 28])
        x = self.func_one(x)
        x = tf.nn.relu(x)
        x = self.func_two(x)
        x = tf.nn.relu(x)
        x = self.func_three(x)
        x = tf.nn.relu(x)
        x = self.func_four(x)
        return x


net_work = MyModel()

net_work.compile(
    optimizer=optimizers.Adam(lr=1e-3),
    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

net_work.fit(db_train, epochs=10, validation_split=0.1,validation_freq=2)

net_work.evaluate(db_test)

# l2_model = keras.Sequential([
#     keras.layers.Dense(512, kernel_regularizer=keras.regularizers.l2(0.001),activation=tf.nn.relu,input_shape=()),
#     keras.layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.001)),
#     keras.layers.Dense(10, activation=tf.nn.sigmoid)
# ])
