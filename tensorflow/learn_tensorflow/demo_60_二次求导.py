import tensorflow as tf

w = tf.Variable(1.0)
x = tf.Variable(2.0)
b = tf.Variable(3.0)

with tf.GradientTape() as tape1:
    with tf.GradientTape() as tape2:
        y = w * x + b
        dy_dw, dy_db = tape2.gradient(y,[w,b])

    dy2_dw2 = tape1.gradient(dy_dw,[w])

print(dy_dw)
print(dy_db)
print(dy2_dw2)



a1 = tf.range(10)
a1 = tf.one_hot(a1,depth=10)
a2 = tf.random.normal([10,10])
# print(a1)
a3 = tf.losses.categorical_crossentropy(a1,a2)

print(a3)
