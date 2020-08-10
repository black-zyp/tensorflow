import tensorflow as tf
from tensorflow.keras import optimizers, datasets, Sequential, metrics, layers
from tensorflow import keras


class MyDense(layers.Layer):
    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()
        self.kernel = self.add_variable('w', [inp_dim, outp_dim])
        self.bias = self.add_variable('b', [outp_dim])

    def call(self, inputs, **kwargs):
        out = inputs @ self.kernel + self.bias
        return out


class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = MyDense(28 * 28, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)

    def call(self, inputs, training=None, mask=None):
        x = self.fc1(inputs)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        x = self.fc3(x)
        x = tf.nn.relu(x)
        x = self.fc4(x)
        x = tf.nn.relu(x)
        x = self.fc5(x)
        return x


def process(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int64)
    x = tf.reshape(x, [28 * 28])
    y = tf.one_hot(y, depth=10)
    return x, y

    # x = tf.cast(x, dtype=tf.float32) / 255.
    # x = tf.reshape(x, [28 * 28])
    # y = tf.cast(y, dtype=tf.int32)
    # y = tf.one_hot(y, depth=10)
    # return x, y


batch_size = 128
(x, y), (x_test, y_test) = datasets.mnist.load_data()

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(process).shuffle(1000).batch(batch_size)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(process).batch(batch_size)

print(x.shape, y.shape)

net_work = MyModel()
net_work.compile(
    optimizer=optimizers.Adam(lr=0.01),
    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
    # loss=tf.losses.MSE,
    metrics=["accuracy"]
)

net_work.fit(db, epochs=5, validation_data=db_test, validation_freq=2)

net_work.evaluate(db_test)
