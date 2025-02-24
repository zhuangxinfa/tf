import os  
import numpy as np  
import tensorflow as tf  
import input_data  
import model  
  
#变量声明  
N_CLASSES = 2 #husky,jiwawa,poodle,qiutian  
IMG_W = 64   # resize图像，太大的话训练时间久  
IMG_H = 64  
BATCH_SIZE =20  
CAPACITY = 200  
MAX_STEP = 100 # 一般大于10K  
learning_rate = 0.0001 # 一般小于0.0001  
  
#获取批次batch  
train_dir = './pic/test_data'   #训练样本的读入路径  
logs_train_dir = './pic/test_data'    #logs存储路径  

train, train_label, val, val_label = input_data.get_files(train_dir, 0.3)  
#训练数据及标签  
train_batch,train_label_batch = input_data.get_batch(train, train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)  
#print(train_label_batch)  #Tensor("Reshape_7:0", shape=(30,), dtype=int32)
#print(train_batch)        #Tensor("batch_9:0", shape=(30, 64, 64, 3), dtype=float32)
#测试数据及标签  
val_batch, val_label_batch = input_data.get_batch(val, val_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)   
  
#训练操作定义  
print(train_batch)
train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)  #inference返回的是一个softmax——linear
#print (train_logits)     ##Tensor("softmax_linear_4/softmax_linear_1:0", shape=(30, 2), dtype=float32)
train_loss = model.losses(train_logits, train_label_batch)    #      
train_op = model.trainning(train_loss, learning_rate)  
train_acc = model.evaluation(train_logits, train_label_batch) #train loss \train op\train acc 是要在sess.run里边进行运行的 
  
#测试操作定义  
test_logits = model.inference(val_batch, BATCH_SIZE, N_CLASSES)  
test_loss = model.losses(test_logits, val_label_batch)          
test_acc = model.evaluation(test_logits, val_label_batch)  
  
#这个是log汇总记录  
summary_op = tf.summary.merge_all()   
  
#产生一个会话  
sess = tf.Session()    
#产生一个writer来写log文件  
train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph) #tensorboard能查看吗？  
#val_writer = tf.summary.FileWriter(logs_test_dir, sess.graph)   
#产生一个saver来存储训练好的模型  
saver = tf.train.Saver()  
#所有节点初始化
sess.run(tf.global_variables_initializer())    
#队列监控  
coord = tf.train.Coordinator()  
threads = tf.train.start_queue_runners(sess=sess, coord=coord)  

#进行batch的训练  
try:  
    #执行MAX_STEP步的训练，一步一个batch  
    for step in np.arange(MAX_STEP): #100
        if coord.should_stop():  #关于多线程停止的一个类
            break  
        #启动以下操作节点，有个疑问，为什么train_logits在这里没有开启？  因为train_logits在train_loss里面开启了
        _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])  
          
        #每10步打印一次当前的loss以及acc，同时记录log，写入writer     
        if step % 1 == 0:  
            print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))  
            summary_str = sess.run(summary_op)  
            train_writer.add_summary(summary_str, step)  
        #保存一次训练好的模型  
        if (step + 1) == MAX_STEP:
            checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')  
            saver.save(sess, checkpoint_path, global_step=step)  
###异常处理         
except tf.errors.OutOfRangeError:  
    print('Done training -- epoch limit reached')  
  
finally:  
    coord.request_stop()  