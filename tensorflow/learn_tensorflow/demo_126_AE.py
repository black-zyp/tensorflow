import tensorflow as tf
from tensorflow.keras import Sequential, metrics, layers, datasets, optimizers
from PIL import Image
from tensorflow import keras
import numpy as np


# Auto_Encoders 的两种变种 1.加噪声 2.dropout

# 图片保存与合并
def save_images(imgs, name):
    new_im = Image.new('L', (280, 280))

    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1

    new_im.save(name)


# 降维的最小维度
h_dim = 20
batch_size = 512
learn_rate = 1e-3

(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()
# 因为是无监督学习  所以自己就是自己的评定标准 所以用不到 y / y_test
x = tf.cast(x, dtype=tf.float32) / 255.
x_test = tf.cast(x_test, dtype=tf.float32) / 255.
db = tf.data.Dataset.from_tensor_slices(x)
db = db.shuffle(batch_size * 5).batch(batch_size)
db_test = tf.data.Dataset.from_tensor_slices(x_test)
db_test = db_test.batch(batch_size)


class AE(keras.Model):
    def __init__(self):
        super(AE, self).__init__()

        # Encoders
        self.encoders = Sequential([
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(64, activation=tf.nn.relu),
            layers.Dense(h_dim)
        ])
        # Decoders
        self.decoder = Sequential([
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(784)
        ])

    def call(self, inputs, training=None):
        x = self.encoders(inputs)
        # 重建
        x_hat = self.decoder(x)

        return x_hat


model = AE()
model.build(input_shape=(None, 784))
model.summary()

loss = metrics.Mean()
acc_loss = metrics.Accuracy()
optimizer = optimizers.Adam(learn_rate)
for epoch in range(100):
    for step, x in enumerate(db):
        x = tf.reshape(x, [-1, 784])
        with tf.GradientTape() as tape:
            x_rec_logits = model(x)

            rec_loss = tf.losses.binary_crossentropy(x, x_rec_logits, from_logits=True)
            loss.update_state(rec_loss)

        grads = tape.gradient(rec_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, 'loss:', loss.result().numpy())
            loss.reset_states()

        # evaluation
        # 在测试集中拿来一张图片
        x = next(iter(db_test))
        # 进行重建  因为 x[b,28,28] model 的输入需要 [b,784]
        logits = model(tf.reshape(x, (-1, 784)))
        # 将重建的参数限制在 0-1 之间
        x_hat = tf.sigmoid(logits)
        # [b,784] => [b,28,28]
        x_hat = tf.reshape(x_hat, [-1, 28, 28])
        # 在 b 维度上 将 x 与 x_hat 进行拼接 [b,28,28] =>> [2b,28,28]
        x_concat = tf.concat([x, x_hat], axis=0)
        x_concat = x_hat
        # 将数据变成 numpy() 类型的数据 放大到 0-255 之间 因为是图片数据
        x_concat = x_concat.numpy() * 255.
        # 转换成 numpy 保存图片的标准格式
        x_concat = x_concat.astype(np.uint8)
        save_images(x_concat, './AE_images/rec_epoch_%d.png' % epoch)
