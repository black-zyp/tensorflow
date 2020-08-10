import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

def himmelblau(x):
    return (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2

x = np.arange(-6,6,0.1)
y = np.arange(-6,6,0.1)
print("x,y range:",x.shape,y.shape)
X,Y = np.meshgrid(x,y)
print("X,Y maps:",X.shape,Y.shape)
Z = himmelblau([X,Y])

fig = plt.figure("himmelblau",figsize=(10,8))
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,Z)
ax.view_init(60,-30)
ax.set_xlabel('x')
ax.set_ylabel('y')
# plt.show()


# [1.,0.] , [-4.,0.] , [4.,0.]
x = tf.constant([-4.,0.])

for step in range(200):
    with tf.GradientTape() as tape:
        tape.watch([x])
        loss = himmelblau(x)
        grad = tape.gradient(loss,[x])[0]
        x -= 0.01 * grad

    if step%10==0:
        print('step{}: x = {} f(x) = {}'.format(step,x.numpy(),loss.numpy()))


