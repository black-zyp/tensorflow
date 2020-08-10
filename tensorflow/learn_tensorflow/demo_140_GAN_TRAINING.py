import os
from io import BytesIO
import numpy as np
import tensorflow as tf
from tensorflow import keras
# from scipy.misc import toimage
from PIL import Image
import glob
from learn_tensorflow.demo_140_GAN import Generator, Discriminator
from learn_tensorflow.demo_140_dataset import make_anime_dataset


def save_result(val_out, val_block_size, image_path, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        # img = img.astype(np.uint8)
        return img

    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b + 1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)

    # toimage(final_image).save(image_path)
    # print(final_image.shape)
    img = Image.fromarray(final_image)
    # .covert('RGB')
    # img.save(image_path)
    buffer = BytesIO()
    img.save(buffer, format="jpeg")
    open(image_path, "wb").write(buffer.getvalue())


def celoss_ones(logits):  # 送的是全为真的数据  labels 就标记为全为 1 为真
    # [b,1]
    # [b] = [1,1,1,1,]
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits))
    return tf.reduce_mean(loss)


def celoss_zeros(fake):  # 送的全是假的数据 labels 就标记为全为 0 为假
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake))
    return tf.reduce_mean(loss)


def gradient_penalty(discriminator, batch_x, fake_image):

    batchsz = batch_x.shape[0]
    t = tf.random.uniform([batchsz, 1, 1, 1])
    t = tf.broadcast_to(t, batch_x.shape)

    interplate = t * batch_x + (1 - t) * fake_image

    with tf.GradientTape() as tape:
        tape.watch([interplate])
        d_interplote_logits = discriminator(interplate)
    grads = tape.gradient(d_interplote_logits, interplate)

    # grads: [b,h,w,c] ==> [b,-1]
    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.norm(grads, axis=1)  # [b]
    gp = tf.reduce_mean((gp - 1) ** 2)
    return gp


def d_loss_function(generator, discriminator, batch_z, batch_x, is_training):
    # 1.treat real image as real
    # 2.treat generate image as fake
    fake_image = generator(batch_z, is_training)  # 假的图片
    d_fake_logits = discriminator(fake_image,is_training)  # 假的图片通过判别器的输出
    d_real_logits = discriminator(batch_x,is_training)  # 真的图片通过判别器的输出

    d_loss_real = celoss_ones(d_real_logits)  # 将真的图片的输出 输入 计算loss  真的样本尽可能接近于 1
    d_loss_fake = celoss_zeros(d_fake_logits)  # 将假的图片的输出 输入 计算loss 假的样本尽可能接均于 0

    gp = gradient_penalty(discriminator, batch_x, fake_image)

    loss = d_loss_fake + d_loss_real + 10. * gp
    return loss, gp


def g_loss_function(generator, discriminator, batch_z, is_training):
    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    loss = celoss_ones(d_fake_logits)
    return loss


def main():
    tf.random.set_seed(22)
    np.random.seed(22)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    assert tf.__version__.startswith('2.')

    z_dim = 100  # 隐藏变量的维度 100 维
    epochs = 3000000
    batch_size = 512
    learning_rate = 0.002
    is_training = True

    # 加载数据集
    image_path = glob.glob(r"C:\Users\SaltedFish\.keras\datasets\faces\faces\*.jpg")
    dataset, img_shape, _ = make_anime_dataset(image_path, batch_size)
    # print(dataset, img_shape)
    sample = next(iter(dataset))
    print(sample, tf.reduce_max(sample), tf.reduce_min(sample))
    # 不设置 repeat 可以无限制的从 dataset 中取样
    dataset = dataset.repeat()
    db_iter = iter(dataset)

    generator = Generator()
    generator.build(input_shape=(None, z_dim))
    discriminator = Discriminator()
    discriminator.build(input_shape=(None, 64, 64, 3))

    g_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    d_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

    for epoch in range(epochs):

        batch_z = tf.random.uniform([batch_size, z_dim], minval=-1., maxval=1.)
        batch_x = next(db_iter)

        # train discriminator  先训练判别器 再固定判别器 训练生成器
        with tf.GradientTape() as tape:
            d_loss, gp = d_loss_function(generator, discriminator, batch_z, batch_x, is_training)

        grads1 = tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(grads1, discriminator.trainable_variables))

        # train generator
        with tf.GradientTape() as tape:
            g_loss = g_loss_function(generator, discriminator, batch_z, is_training)
        grads2 = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads2, generator.trainable_variables))

        if epoch % 100 == 0:
            print(epoch, 'd_loss:', float(d_loss), 'g_loss:', float(g_loss), 'gp:', float(gp))

            z = tf.random.uniform([100, z_dim])
            fake_image = generator(z, training=False)
            # print(fake_image.shape)
            img_path = os.path.join('./save_GAN_images', 'WGAN_%d.jpg' % epoch)
            save_result(fake_image.numpy(), 10, img_path, color_mode='P')


if __name__ == '__main__':
    main()
