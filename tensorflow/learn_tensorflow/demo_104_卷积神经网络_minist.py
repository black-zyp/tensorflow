import tensorflow as tf
from tensorflow.keras import layers, optimizers, Sequential, metrics, datasets
from tensorflow import keras

devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)


def process(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int64)
    return x, y


batch_size = 32
(x, y), (x_test, y_test) = datasets.mnist.load_data()
print(x.shape, y.shape)
db_train = tf.data.Dataset.from_tensor_slices((x, y))
db_train = db_train.shuffle(1000).map(process).batch(batch_size)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(process).batch(batch_size)

sample = next(iter(db_train))
print(sample[0].shape, sample[1].shape)

model = Sequential([
    layers.Conv2D(64, kernel_size=(3, 3), activation=tf.nn.relu, padding='same'),
    layers.Conv2D(64, kernel_size=(3, 3), activation=tf.nn.relu, padding='same'),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    layers.Conv2D(128, kernel_size=(3, 3), activation=tf.nn.relu, padding='same'),
    layers.Conv2D(128, kernel_size=(3, 3), activation=tf.nn.relu, padding='same'),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    layers.Conv2D(256, kernel_size=(3, 3), activation=tf.nn.relu, padding='same'),
    layers.Conv2D(256, kernel_size=(3, 3), activation=tf.nn.relu, padding='same'),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    layers.Conv2D(512, kernel_size=(3, 3), activation=tf.nn.relu, padding='same'),
    layers.Conv2D(512, kernel_size=(3, 3), activation=tf.nn.relu, padding='same'),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    layers.Conv2D(512, kernel_size=(3, 3), activation=tf.nn.relu, padding='same'),
    layers.Conv2D(512, kernel_size=(3, 3), activation=tf.nn.relu, padding='same'),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')
])

fc_model = Sequential([
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(10, activation=None),
])

# x_test = tf.reshape(x_test, [-1, 28, 28, 1])
# print(x.shape, x_test.shape)
model.build(input_shape=[None, 28, 28, 1])
fc_model.build(input_shape=[None, 512])

optimizer = optimizers.Adam(lr=1e-4)
acc_meter = metrics.Accuracy()
loss_meter = metrics.Mean()

# variables =

for epoch in range(50):
    for step, (x, y) in enumerate(db_train):
        with tf.GradientTape() as tape:
            x = tf.reshape(x, [-1, 28, 28, 1])
            out = model(x)
            # print(out.shape)

            out = tf.reshape(out, [-1, 512])
            # print(out.shape)
            logits = fc_model(out)
            y_one_hot = tf.one_hot(y, depth=10)
            # print(y_one_hot.shape)
            # print(logits.shape)
            loss = tf.losses.categorical_crossentropy(y_one_hot, logits, from_logits=True)
            loss_meter.update_state(loss)

        grads = tape.gradient(loss, model.trainable_variables + fc_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables + fc_model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, 'loss:', loss_meter.result().numpy())

    for (x, y) in db_test:
        out = tf.reshape(x, [-1, 28, 28, 1])

        out = model(out)
        # print(out.shape)
        out = tf.reshape(out, [-1, 512])
        logits = fc_model(out)
        prob = tf.nn.softmax(logits, axis=1)
        pred = tf.argmax(prob, axis=1)
        acc_meter.update_state(y, pred)

    print(epoch, "acc result:", acc_meter.result().numpy())
