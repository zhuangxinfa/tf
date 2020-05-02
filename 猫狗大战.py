#将原始图片转换成需要的大小，并将其保存  
#========================================================================================  
import os  
import numpy as np  
import tensorflow as tf    
from PIL import ImageFile  
ImageFile.LOAD_TRUNCATED_IMAGES = True
    
#原始图片的存储位置  
orig_picture = r'C:\Users\zhuan\Desktop\tf\pic'  
  
#生成图片的存储位置  
gen_picture = r'C:\Users\zhuan\Desktop\tf\pic\new'  
  
#需要的识别类型  
classes = {'mao','gou'}   
  
#样本总数  
num_samples = 74   
     
#制作TFRecords数据    
def create_record():    
    writer = tf.python_io.TFRecordWriter("dog_train.tfrecords")    
    for index, name in enumerate(classes):    
        class_path = orig_picture +"/"+ name+"/"    
        for img_name in os.listdir(class_path):    
            img_path = class_path + img_name    
            img = Image.open(img_path)    
            img = img.resize((64, 64))    #设置需要转换的图片大小  
            img_raw = img.tobytes()      #将图片转化为原生bytes    
            print (index,img_raw)    
            example = tf.train.Example(    
               features=tf.train.Features(feature={    
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),    
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))    
               }))    
            writer.write(example.SerializeToString())    
    writer.close()    
      
#=======================================================================================  
def read_and_decode(filename):    
    # 创建文件队列,不限读取的数量    
    filename_queue = tf.train.string_input_producer([filename])    
    # create a reader from file queue    
    reader = tf.TFRecordReader()    
    # reader从文件队列中读入一个序列化的样本    
    _, serialized_example = reader.read(filename_queue)    
    # get feature from serialized example    
    # 解析符号化的样本    
    features = tf.parse_single_example(    
        serialized_example,    
        features={    
            'label': tf.FixedLenFeature([], tf.int64),    
            'img_raw': tf.FixedLenFeature([], tf.string)    
        })    
    label = features['label']    
    img = features['img_raw']    
    img = tf.decode_raw(img, tf.uint8)    
    img = tf.reshape(img, [64, 64, 3])    
    #img = tf.cast(img, tf.float32) * (1. / 255) - 0.5    
    label = tf.cast(label, tf.int32)    
    return img, label    
  
#=======================================================================================  
if __name__ == '__main__':    
    create_record()    
    batch = read_and_decode('dog_train.tfrecords')    
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())    
        
    with tf.Session() as sess: #开始一个会话      
        sess.run(init_op)      
        coord=tf.train.Coordinator()      
        threads= tf.train.start_queue_runners(coord=coord)    
          
        for i in range(num_samples):      
            example, lab = sess.run(batch)#在会话中取出image和label      
            img=Image.fromarray(example, 'RGB')#这里Image是之前提到的   
            img.save(gen_picture+'/'+str(i)+'samples'+str(lab)+'.jpg')#存下图片;注意cwd后边加上‘/’      
            print(example, lab)      
        coord.request_stop()      
        coord.join(threads)     
        sess.close()
  ###########################################################################################
  #############################################################################################
  
        #============================================================================  
#-----------------生成图片路径和标签的List------------------------------------  
  
train_dir = r'C:\Users\zhuan\Desktop\tf\pic\new'  
  
mao = []  
label_mao = []  
gou = []  
label_gou = []  

  
#step1：获取'E:/Re_train/image_data/training_image'下所有的图片路径名，存放到  
#对应的列表中，同时贴上标签，存放到label列表中。  
def get_files(file_dir, ratio):  
    for file in os.listdir(file_dir+'/mao'):  
        husky.append(file_dir +'/mao'+'/'+ file)   
        label_mao.append(0)  
    for file in os.listdir(file_dir+'/gou'):  
        jiwawa.append(file_dir +'/gou'+'/'+file)  
        label_gou.append(1)  
#    for file in os.listdir(file_dir+'/poodle'):  
#        poodle.append(file_dir +'/poodle'+'/'+ file)   
#        label_poodle.append(2)  
#    for file in os.listdir(file_dir+'/qiutian'):  
#        qiutian.append(file_dir +'/qiutian'+'/'+file)  
#        label_qiutian.append(3)  
#  
#step2：对生成的图片路径和标签List做打乱处理把cat和dog合起来组成一个list（img和lab） 
    image_list = np.hstack((mao, gou))      
    #image_list = np.hstack((husky, jiwawa, poodle, qiutian))  
    label_list = np.hstack((label_mao, label_gou))  
  
    #利用shuffle打乱顺序  
    temp = np.array([image_list, label_list])  
    temp = temp.transpose()  
    np.random.shuffle(temp)  
      
    #从打乱的temp中再取出list（img和lab）  
    image_list = list(temp[:, 0])  
    label_list = list(temp[:, 1])  
    label_list = [int(i) for i in label_list]  
    return image_list, label_list  
      
    #将所有的img和lab转换成list  
    all_image_list = list(temp[:, 0])  
    all_label_list = list(temp[:, 1])  
  
    #将所得List分为两部分，一部分用来训练tra，一部分用来测试val  
    #ratio是测试集的比例  
    n_sample = len(all_label_list)  
    n_val = int(math.ceil(n_sample*ratio))   #测试样本数  
    n_train = n_sample - n_val   #训练样本数  
  
    tra_images = all_image_list[0:n_train]  
    tra_labels = all_label_list[0:n_train]  
    tra_labels = [int(float(i)) for i in tra_labels]  
    val_images = all_image_list[n_train:-1]  
    val_labels = all_label_list[n_train:-1]  
    val_labels = [int(float(i)) for i in val_labels]  
  
    return tra_images, tra_labels, val_images, val_labels  
      
      
#---------------------------------------------------------------------------  
#--------------------生成Batch----------------------------------------------  
  
#step1：将上面生成的List传入get_batch() ，转换类型，产生一个输入队列queue，因为img和lab  
#是分开的，所以使用tf.train.slice_input_producer()，然后用tf.read_file()从队列中读取图像  
#   image_W, image_H, ：设置好固定的图像高度和宽度  
#   设置batch_size：每个batch要放多少张图片  
#   capacity：一个队列最大多少  
def get_batch(image, label, image_W, image_H, batch_size, capacity):  
    #转换类型  
    image = tf.cast(image, tf.string)  
    label = tf.cast(label, tf.int32)  
  
    # make an input queue  
    input_queue = tf.train.slice_input_producer([image, label])  
  
    label = input_queue[1]  
    image_contents = tf.read_file(input_queue[0]) #read img from a queue    
      
#step2：将图像解码，不同类型的图像不能混在一起，要么只用jpeg，要么只用png等。  
    image = tf.image.decode_jpeg(image_contents, channels=3)   
      
#step3：数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮。  
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)  
    image = tf.image.per_image_standardization(image)  
  
#step4：生成batch  
#image_batch: 4D tensor [batch_size, width, height, 3],dtype=tf.float32   
#label_batch: 1D tensor [batch_size], dtype=tf.int32  
    image_batch, label_batch = tf.train.batch([image, label],  
                                                batch_size= batch_size,  
                                                num_threads= 32,   
                                                capacity = capacity)  
    #重新排列label，行数为[batch_size]  
    label_batch = tf.reshape(label_batch, [batch_size])  
    image_batch = tf.cast(image_batch, tf.float32)  
    return image_batch, label_batch