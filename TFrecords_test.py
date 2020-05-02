import os  
import tensorflow as tf  
from PIL import Image  
import matplotlib.pyplot as plt  
import numpy as np  
  
cwd = r'C:\Users\zhuan\Desktop\tf\pic'
classes = {'1','2'}  
writer = tf.python_io.TFRecordWriter("train.tfrecords")  
  
for index, name in enumerate(classes):  
    class_path = cwd + name + '\'
    for img_name in os.listdir(class_path):  
        img_path = class_path + img_name  
  
        img = Image.open(img_path)  
        img = img.resize((128,128))  
        img_raw = img.tobytes()  
        example = tf.train.Example(features = tf.train.Features(feature={  
            "label":tf.train.Feature(int64_list = tf.train.Int64List(value=[index])),  
            'img_raw':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw]))  
        }))  
        writer.write(example.SerializeToString())  
  
writer.close()