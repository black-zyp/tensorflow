# import tensorflow as tf
# import os
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# a = tf.constant(2.)
# b = tf.constant(4.)
# print(a * b)

import numpy as np

def sigmoid(X):
    return 1.0/(1+np.exp(-X))

a = np.array([0.5,-1,-5.2,3.6])

result = sigmoid(a)

print(result)



