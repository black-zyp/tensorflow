import tensorflow as tf
import numpy as np
#
# a = tf.constant(1.)
#
# print(a.device)  # device 查看是由 CPU 还是 GPU 创建的 String 类型
# print(type(a))
# p = a.numpy()
#
# # a = a.cpu() # 将 a 转化为在 cpu 下创建
# # a = a.gpu() # 将 a 转化为在 gpu 下创建 不在同一个下创建的不能进行计算
# print(a.ndim)  # 返回数据的行数 返回的是数值型的数据
# print(tf.rank(a))  # 返回数据的行数 返回的是tensor类型的数据
# print(type(p))
#
# # 判断是不是 tensor 的类型
# print(tf.is_tensor(a))
# ss = np.arange(100).reshape(25, 4)
# # 将 numpy 转换成 tensor类型  dtype 指定数据的类型
# tensor_ss1 = tf.convert_to_tensor(ss, dtype=tf.int64)
# tensor_ss2 = tf.cast(ss, dtype=tf.int64)
#
# # 用 Variable 进行包装
# # y = wx + b
# # x,y是tensor类型,w,b是梯度需要优化的参数
# # 经过包装后具有了可以求导的特性
# # 具有了 name(tf1.0中的概念) 和 trainable 的参数 在图中要进行求导
# b = tf.Variable(a, name='a')
#
# ones = tf.ones([2, 3], dtype=tf.int64)  # 建立全为 1 的tensor 第一个参数:几行几列
# zero = tf.zeros([2, 3], dtype=tf.float32)  # 建立全为 0 的tensor 第一个参数:几行几列
# every_num = tf.fill([2, 3], 5)  # 建立全为 5 的tensor 第一个参数:几行几列
# print(ones)
# print(zero)
# print(every_num)

# normal 正态分布 mean 均值 stddev 标准差
ss1 = tf.random.normal([2,2],mean=1,stddev=1)
ss2 = tf.random.normal([2,2])
# 截断的正态分布 truncated_normal 将边缘的无用的数据阶段 在进行重新采样(依然是正太)
ss3 = tf.random.truncated_normal([2,2],mean=0,stddev=1)
print(ss1)
print(ss2)
print(ss3)

# uniform 均匀分布 在 minval 和 maxval 之间进行采样
ss4 = tf.random.uniform([2,2],minval=0,maxval=100)
print(ss4)

# tf.random.shuffle 用来随机打散
ss5 = tf.range(10)
print(ss5)
ss5 = tf.random.shuffle(ss5)
print(ss5)
