import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data  
 
#再入数据库
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

#每个批次的大小
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#初始化权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)#生成一个截断的正态分布
    return tf.Variable(initial)

#初始化偏置
def biases_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#卷积层
def conv2d(x,W):
    #x input tensor of shape '[batch,in_height,in_width,in_channels]'
    #W filter/ kernel tensor of shape[filter_height,fliter_width,in_channels,out_channels]
    #strides[0] = strides[3]=1. strides[1]表示沿着x方向的步长，strides[2]表示沿着y方向的步长
    #padding ‘SAME’，‘VAILD’
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#池化层
def max_pool_2x2(x):
    #ksize[1,x,y,1]
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#定义俩placeholder
x = tf.placeholder(tf.float32,[None,784])#图片大小：28x28
y = tf.placeholder(tf.float32,[None,10])

#改变x的格式为4D的向量[batch,in_height,in_width,in_channels]
#最后一个1表示是黑白图片纬度1，彩色图片用3
x_image = tf.reshape(x,[-1,28,28,1])

#初始化第一个卷基层的权值与偏置
W_convl = weight_variable([5,5,1,32])#5x5的取样窗口，从一个图里面取出32个卷积核
b_convl = biases_variable([32])#每一个卷积核有一个偏置

#把x_image和权值向量进行卷积，再加上偏置值，然后应用relu函数作为激励函数
h_convl = tf.nn.relu(conv2d(x_image,W_convl) + b_convl)
h_pool1 = max_pool_2x2(h_convl)

#进行第二个卷积层权值和偏置的初始化
W_conv2 = weight_variable([5,5,32,64])#5x5的取样窗口，用64个卷积核在32个平面内抽取特征
b_conv2 = biases_variable([64])

#把h_pool1和权值向量进行卷积再加上偏置，然后用relu激励函数
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#初始化第一个全链接层的权值
W_fc1 = weight_variable([7*7*64,1024])
b_fc1= biases_variable([1024])

#把池化层2的输出扁平化为一维
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
#求第一个全连接层的输出
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

#用keep_prob来表示神经元的输出概率
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

#初始化第二个全连接层
W_fc2 = weight_variable([1024,10])
b_fc2 = biases_variable([10])

#计算输出
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

#交叉熵代价函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#使用adamoptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#结果准确率,结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
 
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})
        
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:0.7})
        print("Tter" + str(epoch) + "Testing Accuracy" + str(acc))
