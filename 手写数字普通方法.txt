import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)#read_data_sets定义在mnist.py文件，用于读取数据
import tensorflow as tf#导入tensorflow的包
with tf.name_scope('x'):
    x = tf.placeholder(tf.float32, [None, 784])            #构建占位符，代表输入的图像，None表示样本的数量可以是任意的 
                                                            #第一个参数是要保存的数据类型，第二个参数是要保存的数据结构
                                                            #它在使用的时候和前面的variable不同的是在session运行阶段，
                                                            #需要给placeholder提供数据，利用feed_dict的字典结构给placeholdr变量“喂数据”
                                                            
                                                            #tensorflow中很重要的一个部分就是Variable,它能构建一个变量，在计算图的运算过程中，
                                                            #其值会一直保存到程序运行结束，而一般的tensor张量在tensorflow运行过程中只是在计算图中流过，
                                                            #并不会保存下来，因此varibale主要用来保存tensorflow构建的一些结构中的参数，
                                                            #这样，这些参数才不会随着运算的消失而消失，才能最终得到一个模型。比如神经网络中的权重和bias等，
                                                            #在训练过后，总是希望这些参数能够保存下来，而不是直接就消失了，所以这个时候要用到Variable。
                                                            #注意，所有和varible有关的操作在计算的时候都要使用session会话来控制，包括计算，打印等等。
                                                            
    W = tf.Variable(tf.zeros([784,10]))                    #构建一个变量，代表训练目标 W，初始化为 0  zeros() 创建一个tensor内容全部为零
    b = tf.Variable(tf.zeros([10]))                        #构建一个变量，代表训练目标b，初始化为0  
    y = tf.nn.softmax(tf.matmul(x,W) + b)                  #构建了一个softmax的模型：y = softmax(Wx + b)，y指样本标签的预测值  
y_ = tf.placeholder("float", [None,10])                #构建占位符，代表样本标签的真实值  
cross_entropy = -tf.reduce_sum(y_*tf.log(y))           #交叉熵代价函数  
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)          #使用梯度下降法（0.01的学习率）来最小化这个交叉熵代价函数  
init = tf.initialize_all_variables()                    
sess = tf.Session()                                    #构建会话  
writer = tf.summary.FileWriter("log/",sess.graph)
sess.run(init)                                         #初始化所有变量  
for i in range(1000):                                  #迭代次数为1000  
  batch_xs, batch_ys = mnist.train.next_batch(100)                   #使用minibatch的训练数据，一个batch的大小为100  
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})        #用训练数据替代占位符来执行训练  
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))     #tf.argmax()返回的是某一维度上其数据最大所在的索引值，在这里即代表预测值和真值  
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))    #用平均值来统计测试准确率  
  print (i, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})) #打印测试信息  
sess.close()