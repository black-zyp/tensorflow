import tensorflow as tf
from tensorflow.keras import Sequential, optimizers, layers, metrics, datasets
from tensorflow import keras

# 常见单词的数量
total_words = 10000
batch_size = 128
# 句子的最大长度
max_review_len = 80
embedding_len = 100
(x, y), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)
# 将所有的句子进行一个 padding max_review_len 的长度
x = keras.preprocessing.sequence.pad_sequences(x, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)
# x : [b,80]
# x_test : [b,80]

db_train = tf.data.Dataset.from_tensor_slices((x, y))
db_train = db_train.shuffle(1000).batch(batch_size, drop_remainder=True)  # drop_remainder 将最后的一个 batch 删除掉

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.shuffle(1000).batch(batch_size, drop_remainder=True)

print(x.shape)


class MyRnn(keras.Model):
    def __init__(self, units):
        super(MyRnn, self).__init__()
        # [b,80] ==> [b,80,100]  80个单词,每个单词用 100 维的向量表示
        self.embedding = layers.Embedding(total_words, embedding_len, input_length=max_review_len)
        self.state0 = tf.zeros([batch_size, units])
        self.state1 = tf.zeros([batch_size, units])
        self.state2 = tf.zeros([batch_size, units])
        # [b,80,100] , h_dim:64  语义的提取
        self.rnn_cell0 = layers.SimpleRNNCell(units, dropout=0.2)
        self.rnn_cell1 = layers.SimpleRNNCell(units, dropout=0.2)
        self.rnn_cell2 = layers.SimpleRNNCell(units, dropout=0.2)

        # [b,80,100] => [b.64] => [b,1]
        self.outlayer = layers.Dense(1)

    def call(self, inputs, training=None):
        # inputs:[b,80] ==> [b,80,100]
        x = self.embedding(inputs)
        # rnn cell compute [b,80,100] => [b,64]
        state0 = self.state0
        state1 = self.state1
        state2 = self.state2
        for word in tf.unstack(x, axis=1):
            #
            out0, state0 = self.rnn_cell0(word, state0, training)
            out1, state1 = self.rnn_cell1(out0, state1, training)
            out2, state2 = self.rnn_cell2(out1, state2, training)

        # out:[b,64] =>[b,1]
        x = self.outlayer(out2)
        # 0-1
        prob = tf.sigmoid(x)
        return prob


def main():
    # units = 64
    # epoch = 20
    # model = MyRnn(units)
    # model.compile(optimizer=keras.optimizers.Adam(0.001),
    #               loss=tf.losses.BinaryCrossentropy(),
    #               metrics=['accuracy'])
    #
    # model.fit(db_train, epochs=epoch, validation_data=db_test)
    #
    # model.evaluate(db_test)

    # 梯度的裁剪  将梯度限制在 15(10左右) 以内 如果超过 15 就进行 g/|g|+15
    # grad = [tf.clip_by_norm(g,15) for g in grads]

    units = 64

    acc_loss = metrics.Accuracy()
    loss_meter = metrics.Mean()
    optimizer = optimizers.Adam(1e-3)

    for epoch in range(20):
        for step, (x, y) in enumerate(db_train):
            with tf.GradientTape() as tape:
                logits = MyRnn(units)
                print(type(logits))
                # y_one_hot = tf.one_hot(y, depth=1)
                # print(y_one_hot.shape,type(y_one_hot))
                loss = tf.losses.categorical_crossentropy(y, logits, from_logits=True)
                loss_meter.update_state(loss)

            grads = tape.gradient(loss, logits.trainable_variables)
            optimizer.apply_gradients(zip(grads, logits.trainable_variables))

            if step % 100 == 0:
                print(epoch,step,'loss:',loss_meter.result().numpy())



if __name__ == '__main__':
    main()
