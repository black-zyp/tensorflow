import tensorflow as tf
from tensorflow.keras import datasets, optimizers, metrics, layers, Sequential

(x, y), (x_test, y_test) = datasets.mnist.load_data()


def process(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int64)
    return x, y


db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(process).shuffle(60000).batch(128)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(process).batch(128)

net_work = Sequential([
    layers.Reshape(target_shape=(-1,512)),
    layers.Dense(512, activation=tf.nn.relu),
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(10, activation=tf.nn.relu),
])

net_work.build([None, 28 * 28])
net_work.summary()

# 简单写法(还未成功)
net_work.compile(
    optimizer=optimizers.Adam(lr=0.01),
    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

net_work.fit(db,epochs=10,validation_data=db_test,validation_freq=2)

net_work.evaluate(db_test)


# # 常规写法
# optimizer = optimizers.Adam(lr=1e-3)
# acc_meter = metrics.Accuracy()
# loss_meter = metrics.Mean()
#
# for epoch in range(30):
#     for step, (x, y) in enumerate(db):
#         x = tf.reshape(x, [-1, 28 * 28])
#         with tf.GradientTape() as tape:
#             logic = net_work(x)
#             y_one_hot = tf.one_hot(y, depth=10)
#             loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_one_hot, logic, from_logits=True))
#             loss_meter.update_state(loss)
#         grad = tape.gradient(loss, net_work.trainable_variables)
#         optimizer.apply_gradients(zip(grad, net_work.trainable_variables))
#
#             if step % 100 == 0:
#                 print(epoch, step, "loss:", loss_meter.result().numpy())
#                 loss_meter.reset_states()
#
#     acc_meter.reset_states()
#
#     for step, (x, y) in enumerate(db_test):
#         x = tf.reshape(x, [-1, 28 * 28])
#         with tf.GradientTape() as tape:
#             logic = net_work(x)
#             pred = tf.nn.softmax(logic, axis=1)
#             pred = tf.argmax(pred, axis=1)
#             acc_meter.update_state(y, pred)
#
#     print(epoch, "acc_meter:", acc_meter.result().numpy())
