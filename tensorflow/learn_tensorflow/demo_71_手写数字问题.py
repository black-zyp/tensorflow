import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from tensorflow import keras


# 数据的处理函数,将数据的类型进行转换
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


# 1. 加载数据集
(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()
print(x.shape,y.shape)
batch_size = 128
# 2.构造数据集,直接将 x,y 转换成 tensor 格式
db = tf.data.Dataset.from_tensor_slices((x, y))
# 3.先将数据集用处理函数处理一下,让后用 shuffle 打乱 , 最后设置 batch 每一次取 128 张图片
db = db.map(preprocess).shuffle(10000).batch(batch_size)

# 4.构造测试集
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(batch_size)

db_iter = iter(db)
sample = next(db_iter)

# 5.构造模型(网络) Sequential 是一个容器
model = Sequential([
    layers.Dense(256, activation=tf.nn.relu),  # 第一层[b,784] => [b,256]
    layers.Dense(128, activation=tf.nn.relu),  # 第二层[b,256] => [b,128]
    layers.Dense(64, activation=tf.nn.relu),  # 第三层[b,128] => [b,64]
    layers.Dense(32, activation=tf.nn.relu),  # 第四层[b,64] => [b,32]
    layers.Dense(10)  # 第五层[b,32] => [b,10]
])
# 6.设置输入的维度 [自动匹配,784]
model.build(input_shape=[None, 28 * 28])
# 这一句话是神经网络的详细信息
model.summary()

# 构造优化器 传入的值是 list
optimizer = optimizers.Adam(lr=1e-3)


def main():
    for epoch in range(30):
        for step, (x, y) in enumerate(db):
            # 7.将 x 进行 reshape => [吧,784]
            x = tf.reshape(x, [-1, 28 * 28])
            # 8.进行前向传播的自动求导
            with tf.GradientTape() as tape:
                logits = model(x)
                y_onehot = tf.one_hot(y, depth=10)
                loss_mse = tf.reduce_mean(tf.losses.MSE(y_onehot, logits))
                loss_ce = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True))
            # model.trainable_variables 返回成员变量 相当于:[w1, b1, w2, b2, w3, b3]
            grads = tape.gradient(loss_ce, model.trainable_variables)
            # 进行原地更新  zip 是将 grads 和 model.trainable_variables 的第 0 个元素进行结合
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, "loss:", float(loss_ce), float(loss_mse))
        # # test
        # total_correct = 0  # 在预测值中与真实值相等样本的个数
        # total_num = 0  # 总共进行多少样本的计算数
        # for x, y in db_test:
        #     x = tf.reshape(x, [-1, 28 * 28])
        #     # 利用设计好的模型进行前向传播计算
        #     logits = model(x)
        #     # 将预测值 映射到[0-1]之间,并且和为 1 ,在 1 维度上操作
        #     prob = tf.nn.softmax(logits, axis=1)
        #     # 取出每一个样本中的最大的预测值的下标,归为一类(总共10类)
        #     pred = tf.argmax(prob, axis=1)
        #     pred = tf.cast(pred, tf.int32)
        #     # 计算两个 tensor 中相等的,返回一个 tensor
        #     correct = tf.equal(pred, y)
        #     # 将其进行 类型转换 和 求和  返回一个一维 的 tensor
        #     correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))
        #     total_correct += int(correct)
        #     total_num += x.shape[0]
        # acc = total_correct / total_num
        # print(epoch, "test_acc:", acc)


if __name__ == '__main__':
    main()
