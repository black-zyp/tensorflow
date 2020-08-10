import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets

# import os
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# x : [60k,28,28],
# y : [60k]
# 1. 加载数据集
(x, y), (x_test, y_test) = datasets.mnist.load_data()

# 2. 将数据转化成一个 tensor
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)

x_test = tf.convert_to_tensor(x_test, dtype=tf.float32) / 255.
y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

print(x.shape, y.shape, x.dtype, y.dtype)
print(tf.reduce_min(x), tf.reduce_max(x))
print(tf.reduce_min(y), tf.reduce_max(y))

# 每一次取 128 张图片 总共 60000 张
train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)
# 创建训练迭代器,将数据设置为可以迭代
train_iter = iter(train_db)
# 调用 next 不停的迭代 获取每一个元素 [128,28,28]
sample = next(train_iter)

print(sample[0].shape, sample[1].shape)

# [b,784] => [b,256] => [b,128] => [b,10]
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

lr = 1e-3  # 0.001  10 的 -3 次方  1e-3
for epoch in range(10):  # epoch 是针对数据集的循环 将数据及循环 10 次
    for step, (x, y) in enumerate(train_db):  # step 是针对 batch 的循环
        # x : [128,28,28]
        # y : [128]

        # [b,28,28] => [b,28*28]
        x = tf.reshape(x, [-1, 28 * 28])

        with tf.GradientTape() as tape:  # 只会跟踪 tf.Variable 的类型
            # x : [b,28*28]
            # h1 = x @ W1 + b1
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            out = h2 @ w3 + b3

            # 计算误差 Loss 函数
            # out : [b,10]
            # y : [b]  将 y 进行 one_hot
            y_one_hot = tf.one_hot(y, depth=10)

            # mse = mean(sum(y-out)**2)
            # [b,10]
            loss = tf.square(y_one_hot - out)
            # mean:scala
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # print("--before--")
        # for i in grads:
        #     print(tf.norm(i))

        # grads, _ = tf.clip_by_global_norm(grads, 15)
        # print("--after--")
        # for i in grads:
        #     print(tf.norm(i))


        # w1 = w1 - lr * w1_grad
        # 原地更新
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])
        # w1 = w1 - lr * grads[0]
        # b1 = b1 - lr * grads[1]
        # w2 = w2 - lr * grads[2]
        # b2 = b2 - lr * grads[3]
        # w3 = w3 - lr * grads[4]
        # b3 = b3 - lr * grads[5]

        if step % 100 == 0:
            print(epoch, step, 'Loss:', float(loss))

    total_correct, total_num = 0, 0
    for step, (x, y) in enumerate(test_db):
        # [b,28,28] =>> [b,28*28]
        x = tf.reshape(x, [-1, 28 * 28])
        # [b,784] => [b,256] => [b,128] => [b,10]
        h1 = tf.nn.relu(x @ w1 + b1)
        h2 = tf.nn.relu(h1 @ w2 + b2)
        out = h2 @ w3 + b3

        # out: [b,10]
        # 将实数范围的值映射到 0-1 的范围内 概率之和为 1
        prob = tf.nn.softmax(out, axis=1)
        # 选取概率最大的值所在的下标
        pred = tf.argmax(prob, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)
        # y:[b]
        correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
        correct = tf.reduce_sum(correct)

        total_correct += int(correct)
        total_num += x.shape[0]

    acc = total_correct / total_num
    print("acc number:", acc)
