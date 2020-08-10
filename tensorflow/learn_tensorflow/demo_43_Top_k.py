import tensorflow as tf

def accuracy(out_put,target,topk=(1,)):
    # 选取最大的 k 值
    maxk = max(topk)
    # 这是样本 数量
    batch_size = target.shape[0]
    # 将最大值的索引作为预测值,并进行转置
    pred = tf.math.top_k(out_put,maxk).indices
    pred = tf.transpose(pred,perm=[1,0])
    # 对目标值进行维度扩张,扩张成和预测值一样的维度
    target_ = tf.broadcast_to(target,pred.shape)
    # [10,b]  # 将两个唯独不相等的 tensor 进行比较, 相等的标记为 True ,不相等的标记为 False
    correct = tf.equal(pred,target_)
    # 新建一个列表,用于存储预测值
    res = []
    for k in topk:
        # 将这个由 True 和 False 组成的 tensor 转化为 一维的 tensor  True=1  False=0  k 为几就选取几个
        correct_k = tf.cast(tf.reshape(correct[:k],[-1]),dtype=tf.float32)
        # 将 tensor 进行求和 就得到一个标量,再算概率
        correct_k = tf.reduce_sum(correct_k)
        acc = float(correct_k * (100.0 / batch_size))
        # 将得出的概率存到列表中
        res.append(acc)

    return res

if __name__ == '__main__':
    # 初始化 10 个样本, 总共有 6 类 ,6 列就是属于每个类别的概率
    out_put = tf.random.normal([10,6])
    # 将样本规定为 所有类别的概率总和为 1
    out_put = tf.math.softmax(out_put,axis=1)
    # 设计目标值 均匀分布 
    target = tf.random.uniform([10],maxval=6,dtype=tf.int32)
    print('prob:',out_put.numpy())
    pred = tf.argmax(out_put,axis=1)
    print('pred:',pred.numpy())
    print('lable:',target.numpy())

    acc = accuracy(out_put,target,topk=(1,2,3,4,5,6))
    print('top1-6',acc)
