import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

a = tf.constant(10)

b = tf.constant(12)

print(a + b)
