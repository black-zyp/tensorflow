import tensorflow as tf
import numpy as np

ss = np.array([1,2,3,4,5,6])
one = tf.convert_to_tensor(ss,dtype=tf.float32)
ss1 = tf.zeros([2,3])
ss2 = tf.ones([3,3,3])
ss3 = tf.fill([3,3],9.0)
ss4 = tf.random.normal([2,2],mean=1,stddev=1)
ss5 = tf.random.truncated_normal([2,2],mean=1,stddev=1)
ss6 = tf.random.uniform([2,2],minval=0,maxval=0.5)
ss7 = tf.random.shuffle(ss)
out = tf.random.uniform([4,10])
y = tf.range(4)
y = tf.one_hot(y,depth=10)
loss = tf.keras.losses.mse(y,out)
print(out)
print(y)
print(loss)
print(ss7)
# print(ss6)
# print(ss5)
# print(ss4)
# print(ss)
print(one)
# print(ss1)
# print(ss2)
# print(ss3)
print("*"*100)
ss8 = tf.random.normal([4,28,28,3])
print(ss8[1,27,12])

a = tf.range(10)
print(a[-1])
print(a[-1:])

print("*"*100)
ss9 = tf.random.normal([4,35,8])
ss9_select = tf.gather(ss9,axis=1,indices=[2,3])
print(ss9_select)

print("*"*100)
a = tf.random.normal([4,3,2])
aa1 = tf.gather_nd(a,[[0,1,1]])
aa2 = tf.boolean_mask(a,mask=[True,False,True],axis=1)
print(aa2)
print(a.ndim)

print("*"*100)
b1 = tf.random.normal([4,28,28,3])
b2 = tf.reshape(b1,[4,-1])
print(b2)

print("*"*100)
c1 = tf.random.normal([33,22,11,44])
# [0,1,2,3]
# c2 = tf.reshape(c1,[3,4])
print(c1)
c3 = tf.transpose(c1,perm=[0,1,3,2])
print(c3)

print("*"*100)
d1 = tf.constant([[1,2,6],[5,3,4]],dtype=tf.float32)
d2 = tf.tile(d1,[6,1])
print(d2)

print("*"*100)
f1 = tf.range(12)
f2 = tf.reshape(f1,[3,4])
print(f2)
f3 = tf.expand_dims(f2,axis=0)
print(f3)
f4 = tf.squeeze(f3,axis=0)
print(f4)

print("*"*100)
g1 = tf.random.uniform([4,2,3])
g2 = tf.fill([3,5],0.5)

g3 = tf.broadcast_to(g2,[4,3,5])
g4 = tf.matmul(g1,g3)
g5 = g1 @ g3
print(g4.shape)
print(g5.shape)

print("*"*100)
h1 = tf.fill([3,2],2)
h2 = tf.fill([2,3],2)
h3 = tf.constant(1)
out = h1@h2+h3
out_finally = tf.nn.relu(out)
print(out_finally)


print("*"*100)
x = tf.random.normal([2,3,4])
print(x)
print()
sss = tf.transpose(x,perm=[1,0,2])
print(sss)


