import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import Sequential, layers, optimizers, datasets, metrics
import io
import datetime

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float64) / 255.
    y = tf.cast(y, dtype=tf.int64)
    return x, y


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def image_grid(images):
    """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(10, 10))
    for i in range(25):
        # Start next subplot.
        plt.subplot(5, 5, i + 1, title='name')
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)

    return figure


(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()

batch_size = 128
db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(10000).batch(batch_size)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(batch_size)

model = Sequential([
    layers.Dense(512, activation=tf.nn.relu),
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(10)
])

model.build(input_shape=[None, 28 * 28])
optimizer = optimizers.Adam(1e-3)
acc_meter = metrics.Accuracy()
loss_meter = metrics.Mean()

for epoch in range(30):
    for step, (x, y) in enumerate(db):
        x = tf.reshape(x, [-1, 784])
        with tf.GradientTape() as tape:
            logits = model(x)
            y_one_hot = tf.one_hot(y, depth=10)
            loss_ce = tf.reduce_mean(tf.losses.categorical_crossentropy(y_one_hot, logits, from_logits=True))
            loss_meter.update_state(loss_ce)
        grads = tape.gradient(loss_ce, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, 'loss:', loss_meter.result().numpy())
            loss_meter.reset_states()

    # test
    acc_meter.reset_states()
    for x, y in db_test:
        x = tf.reshape(x, [-1, 28 * 28])
        logit = model(x)
        prob = tf.nn.softmax(logit, axis=1)
        pred = tf.argmax(prob, axis=1)
        acc_meter.update_state(y, pred)

    print(epoch, "acc_meter:", acc_meter.result().numpy())
