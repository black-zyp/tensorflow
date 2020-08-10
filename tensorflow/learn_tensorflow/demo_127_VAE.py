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

# (x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()
# # 因为是无监督学习  所以自己就是自己的评定标准 所以用不到 y / y_test
# x = x.astype(np.float32)/ 255.
# x_test = x_test.astype(np.float32) / 255.
# db = tf.data.Dataset.from_tensor_slices(x)
# db = db.shuffle(batch_size * 5).batch(batch_size)
# db_test = tf.data.Dataset.from_tensor_slices(x_test)
# db_test = db_test.batch(batch_size)

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.
# we do not need label
db = tf.data.Dataset.from_tensor_slices(x_train)
db = db.shuffle(batch_size * 5).batch(batch_size)
db_test = tf.data.Dataset.from_tensor_slices(x_test)
db_test = db_test.batch(batch_size)


z_dim = 10


class VAE(keras.Model):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoders
        self.fc1 = layers.Dense(128)
        # 用来得到 均值  fc1 => fc2
        self.fc2 = layers.Dense(z_dim)
        # 用来得到 方差  fc1 => fc3
        self.fc3 = layers.Dense(z_dim)

        # Decoders
        self.fc4 = layers.Dense(128)
        self.fc5 = layers.Dense(784)

    # 解码传播过程
    def encoder(self, x):
        h = tf.nn.relu(self.fc1(x))
        mu = self.fc2(h)
        log_var = self.fc3(h)

        return mu, log_var

    # 重新编码传播过程
    def decoder(self, z):
        out = tf.nn.relu(self.fc4(z))
        out = self.fc5(out)
        return out

    # 这是保持 z 的分布不变 同时又可导的变形
    def reparameterize(self, mu, log_var):
        # 正态分布
        eps = tf.random.normal(log_var.shape)
        std = tf.exp(log_var) ** 0.5
        z = mu + eps * std
        return z

    def call(self, inputs, training=None):
        # [b,784] ==> [b,d_dim]  [b,d_dim]
        mu, log_var = self.encoder(inputs)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var


model = VAE()
model.build(input_shape=(4,784))
# model.build(input_shape=(4, 784))
optimizer = optimizers.Adam(learn_rate)

for epoch in range(1000):
    for step, x in enumerate(db):
        x = tf.reshape(x,[-1,784])
        with tf.GradientTape() as tape:
            x_rec_logits, mu, log_var = model(x)
            rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_rec_logits)
            rec_loss = tf.reduce_sum(rec_loss)/ x.shape[0]

            # 计算 KL 散度  在分布 N(0~1)
            kl_div = -0.5 * (log_var + 1 - mu ** 2 - tf.exp(log_var))
            kl_div = tf.reduce_sum(kl_div) / x.shape[0]
            loss = rec_loss + 1. * kl_div

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 200 == 0:
            print(epoch, step, "kl_div:",float(kl_div),"loss:",float(rec_loss))


    # # evaluation
    # z = tf.random.normal((batch_size, z_dim))
    # logits = model.decoder(z)
    # x_hat = tf.sigmoid(logits)
    # x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() *255.
    # x_hat = x_hat.astype(np.uint8)
    # save_images(x_hat, 'vae_images/sampled_epoch%d.png'%epoch)


    x = next(iter(db_test))
    x = tf.reshape(x,[-1,784])
    logits,_,_ = model(x)
    x_hat = tf.sigmoid(logits)
    x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() *255.
    x_hat = x_hat.astype(np.uint8)
    save_images(x_hat, 'vae_images/rec_epoch%d.png'%epoch)

