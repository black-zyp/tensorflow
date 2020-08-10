import tensorflow as tf

# # 分割与合并
# a = tf.ones([4, 35, 8])
# b = tf.zeros([4, 35, 8])
# result1 = tf.concat([a, b], axis=0)
# result2 = tf.stack([a, b], axis=2)
# result3 = tf.split(result2, axis=3, num_or_size_splits=[2, 2, 4])
# print(result1.shape)
# print(result2.shape)
# print(result3[0].shape, result3[1].shape, result3[2].shape)

# 排序
# a1 = tf.random.shuffle(tf.range(5))
# print(a1)  # [3,1,2,0,4]
# a2 = tf.sort(a1, direction='DESCENDING')
# print(a2)  # [4,3,2,1,0]
# a3 = tf.argsort(a1, direction='DESCENDING')
# print(a3)  # [4,0,2,1,3]
# a4 = tf.gather(a1, a3)
# print(a4)
#
# # Top_k accuracy
# prob = tf.constant([[0.1, 0.2, 0.7], [0.2, 0.7, 0.1]])
# target = tf.constant([2, 0])
# prob1 = tf.reshape(prob[:1], [-1])
# print(prob)
# print(prob[:3])
# print(prob1)
#
# # 填充与复制
# b1 =tf.reshape(tf.range(9.),[3,3])
# 填补, 第二个参数[[上,下],[左,右]]
# b2 = tf.pad(b1,[[1,1],[1,1]],constant_values=10.)
# print(b2)
#
# # 张量限幅
# c1 = tf.random.normal([3,3],mean=20,stddev=2)
# c2 = tf.norm(c1)
# print(c2)
# c3 = tf.clip_by_norm(c1,40)
# c3 = tf.norm(c3)
# print(c3)
#
# # where
# d1 = tf.random.normal([3,3])
# print(d1)
# d2 = d1>0
# print(d2)
# d3 = tf.boolean_mask(d1,d2)
# print(d3)
# # 显示出值为 True 的坐标
# d4 = tf.where(d2)
# print(d4)
# d5 = tf.gather_nd(d1,d4)
# print(d5)

# # meshgrid
# e1 = tf.linspace(-2.0,2.0,5)
# print("e1:",e1)
# e2 = tf.linspace(-2.0,2.0,5)
# point_x , point_y = tf.meshgrid(e1,e2)
# print("point_x:",point_x)
#
# total = tf.stack([point_x,point_y],axis=2)
# print(total)

#
# prob = tf.constant([[0.1, 0.2, 0.7], [0.2, 0.7, 0.1]])
# target = tf.constant([2, 0])
#
# k_b = tf.math.top_k(prob, k=2)
#     # .indices
#
# target = tf.broadcast_to(target, [2,2])
#
# print(prob)
# print(target)
# print(k_b)


# prob = tf.constant([[0.1, 0.2, 0.7], [0.2, 0.7, 0.1]])
#
# x = tf.math.top_k(prob,2).indices
#
# print(prob)
# print(x)

a = (1,2,3)
print(type(1/2))
