import tensorflow as tf
from tensorflow.keras import datasets, Sequential, optimizers, layers, metrics
from tensorflow import keras


def process(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int64)
    y = tf.one_hot(y, depth=10)
    return x, y


batch_size = 128
(x, y), (x_test, y_test) = datasets.cifar10.load_data()
y = tf.squeeze(y)
y_test = tf.squeeze(y_test)
db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(process).shuffle(10000).batch(batch_size)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(process).batch(batch_size)


class MyDense(layers.Layer):
    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()
        self.kernel = self.add_variable('w', [inp_dim, outp_dim])
        self.bais = self.add_variable('b',[outp_dim])

    def call(self, inputs, **kwargs):
        x = inputs @ self.kernel
        return x


class MyNetWork(keras.Model):
    def __init__(self):
        super(MyNetWork, self).__init__()
        self.fc1 = MyDense(32 * 32 * 3, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)

    def call(self, inputs, training=None, mask=None):
        inputs = tf.reshape(inputs, [-1, 32 * 32 * 3])
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


net_work = MyNetWork()
net_work.compile(
    optimizer=optimizers.Adam(lr=1e-3),
    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

net_work.fit(db, epochs=5, validation_data=db_test, validation_freq=1)

net_work.evaluate(db_test)

# 模型保存 ---- 只保存所有的参数
net_work.save_weights('./save_model/cifar10_weights_part')
del net_work
print("save cifar10_weights model")

net_work = MyNetWork()
net_work.compile(
    optimizer=optimizers.Adam(lr=1e-3),
    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)
net_work.load_weights('./save_model/cifar10_weights_part')
print("load weight from file:")
net_work.evaluate(db_test)

# 模型保存 ---- 保存所有细节(暴力保存,性能低)
# net_work.save("./save_model/cifar10_all")
# del net_work
# new_net_work = tf.keras.models.load_model("./save_model/cifar10_all")
# new_net_work.evaluate()
#
# # 模型保存 ---- 工业化保存
# tf.saved_model.save(net_work,"./save_model/cifar10_industry")
# imported = tf.saved_model.load("./save_model/cifar10_industry")
# function_model = imported.signatures['saving_default']
# print(function_model(db_test))





