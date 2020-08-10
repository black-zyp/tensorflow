import tensorflow as tf
from tensorflow.keras import datasets

# 1.加载数据
(x, y), (x_test, y_test) = datasets.mnist.load_data()

# 2.将数据转化成一个 tensor
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int64)

x_test = tf.convert_to_tensor(x_test, dtype=tf.float32) / 255.
y_test = tf.convert_to_tensor(y_test, dtype=tf.int64)

train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)
next(iter(train_db))

# 设置 w1,b1,w2,b2,w3,b3
# [b,784] => [b,256] => [b,128] => [b,10]
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

lr = 1e-3
for epoch in range(100):
    for step, (x, y) in enumerate(train_db):
        x = tf.reshape(x, [-1, 28 * 28])
        with tf.GradientTape() as tape:
            h1 = tf.nn.relu(x @ w1 + b1)
            h2 = tf.nn.relu(h1 @ w2 + b2)
            out = tf.nn.relu(h2 @ w3 + b3)
            y_one_hot = tf.one_hot(y, depth=10)
            loss = tf.square(y_one_hot - out)
            loss = tf.reduce_mean(loss)

            grad = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])

            # grad, _ = tf.clip_by_global_norm(grad, 15)

            w1.assign_sub(lr * grad[0])
            b1.assign_sub(lr * grad[1])
            w2.assign_sub(lr * grad[2])
            b2.assign_sub(lr * grad[3])
            w3.assign_sub(lr * grad[4])
            b3.assign_sub(lr * grad[5])

        if step % 100 == 0:
            print(epoch, step, 'Loss:', float(loss))

    total_num, total_correct = 0, 0
    for step, (x, y) in enumerate(test_db):
        x = tf.reshape(x, [-1, 28 * 28])
        h1 = tf.nn.relu(x @ w1 + b1)
        h2 = tf.nn.relu(h1 @ w2 + b2)
        out = tf.nn.relu(h2 @ w3 + b3)
        # 映射到 0-1 范围内
        prob = tf.nn.softmax(out, axis=1)
        pred = tf.argmax(prob, axis=1)
        correct = tf.cast(tf.equal(pred, y), dtype=tf.int64)
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_num += x.shape[0]

    acc = total_correct / total_num
    print("acc number:", acc)
