from PIL import Image  
import numpy as np  
import tensorflow as tf  
import matplotlib.pyplot as plt  
import model  
from input_data import get_files  
  
#=======================================================================  
#获取一张图片  
def get_one_image(train):  #train在main函数里边实参是：val2有16个地址正好是56的0.3倍是图片的地址
#    例如：'C:/Users/zhuan/Desktop/tf/pic/test_data/gou/40samples1.jpg
    #输入参数：train,训练图片的路径  
    #返回参数：image，从训练图片中随机抽取一张图片  
    n = len(train)  #n是整形正好是16个
   
    ind = np.random.randint(0, n)  
    img_dir = train[ind]   #随机选择测试的图片  的地址
  
    img = Image.open(img_dir)  
    plt.imshow(img)  #在屏幕上显示图片
    imag = img.resize([64, 64])  #由于图片在预处理阶段以及resize，因此该命令可略  
#    print(imag1)
    image = np.array(imag)  
#    print (image)#打印图片的话就是打印出图片的矩阵
    return image #返回一张图片 
  
#--------------------------------------------------------------------  
#测试图片  
def evaluate_one_image(image_array):  #接收上一个函数返回的image
    with tf.Graph().as_default():  
       BATCH_SIZE = 1 
#       N_CLASSES = 4         
       N_CLASSES = 2
       x = tf.placeholder(tf.float32, shape=[64, 64, 3]) 
       image = tf.cast(x, tf.float32)  #把图片转换成float类型
       image = tf.image.per_image_standardization(image)  #图片标准化
       image = tf.reshape(image, [1, 64, 64, 3])  
  
       logit = model.inference(image, BATCH_SIZE, N_CLASSES)  #inference 返回一个  softmax_linear  
#       print(logit,123456789)
       
  
       logit = tf.nn.softmax(logit)  
  
#       x = tf.placeholder(tf.float32, shape=[64, 64, 3])  
  
       # you need to change the directories to yours.  
#       logs_train_dir = 'C:/Users/zhuan/Desktop/tf/pic/input_data' 
       logs_train_dir = 'C:/Users/zhuan/Desktop/tf/pic/test_data' 

#       logs_train_dir = 'E:/Re_train/image_data/inputdata/'  
  
       saver = tf.train.Saver()  
  
       with tf.Session() as sess:  
  
           print("Reading checkpoints...")  
           ckpt = tf.train.get_checkpoint_state(logs_train_dir)  
           if ckpt and ckpt.model_checkpoint_path:  
               global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]  
               saver.restore(sess, ckpt.model_checkpoint_path)  
               print('Loading success, global_step is %s' % global_step)  
           else:  
               print('No checkpoint file found')  
  
           prediction = sess.run(logit, feed_dict={x: image_array}) #logit 会返回一个soft_liner保存了概率
#           print(sess.run(logit))
#           print(prediction,123456789)
           max_index = np.argmax(prediction)  #获得这个list中最大的数的位置
           if max_index==0:  
               print('This is a 无缺陷 with possibility %.6f' %prediction[:, 0])  
           elif max_index==1:  
               print('This is a 有缺陷 with possibility %.6f' %prediction[:, 1])  
#           elif max_index==2:  
#               print('This is a poodle with possibility %.6f' %prediction[:, 2])  
#           else:  
#               print('This is a qiutian with possibility %.6f' %prediction[:, 3])  
#  
#------------------------------------------------------------------------  
                 
if __name__ == '__main__':  
      
#    train_dir = r'C:/Users/zhuan/Desktop/tf/pic/input_data' 
    testdir = r'C:/Users/zhuan/Desktop/tf/pic/test_data/test' 
#    train, train_label, val, val_label  = get_files(train_dir, 0.3)
    train1,train_label2,val2,val_label2 = get_files(testdir,1.0)
#    print(val2)
    
   
    img = get_one_image(val2)  #通过改变参数train or val，进而验证训练集或测试集  
    evaluate_one_image(img)