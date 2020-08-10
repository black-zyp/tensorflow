import tensorflow as tf
from tensorflow.keras import Sequential, layers, datasets, metrics, optimizers
# from learn_tensorflow.demo_104_resNet_cifar100 import resnet18
from learn_tensorflow.demo_104_resnet18_copy import resnet18

devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)


def process(x, y):
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int64)
    return x, y


(x, y), (x_test, y_test) = datasets.cifar100.load_data()
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)

batch_size = 256
db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(process).shuffle(10000).batch(batch_size)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(process).batch(batch_size)


def main():
    model = resnet18()
    model.build(input_shape=(None, 32, 32, 3))
    optimizer = optimizers.Adam(lr=1e-3)
    acc_meter = metrics.Accuracy()
    loss_meter = metrics.Mean()

    for epoch in range(50):
        for step, (x, y) in enumerate(db):
            with tf.GradientTape() as tape:
                logits = model(x)
                y_one_hot = tf.one_hot(y, depth=100)
                loss = tf.losses.categorical_crossentropy(y_one_hot, logits, from_logits=True)
                loss_meter.update_state(loss)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, 'loss:', loss_meter.result().numpy())

        for x_test, y_test in db_test:
            logits = model(x_test)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            acc_meter.update_state(y_test, pred)

        print(epoch, "acc_meter:", acc_meter.result().numpy())


if __name__ == '__main__':
    main()
