from random import shuffle
import glob
import os, sys
import config
import tensorflow as tf
import cv2
import numpy as np
from tqdm import tqdm
DATASET = config.XDATASET_DIR

training_list = list()
labels_list = list()
for foldNum in os.listdir(DATASET):
    for category in os.listdir(DATASET+foldNum):
        for images in os.listdir(DATASET+foldNum+"/"+category):
            if(category == 'all'):
                training_list.append(DATASET+foldNum+"/"+category+"/"+images)
                labels_list.append(1)
            else:
                training_list.append(DATASET+foldNum+"/"+category+"/"+images)
                labels_list.append(0)

path = './tfrecords/'
if not os.path.isdir('./tfrecords'):
    print("creating tfrecords folder")
    os.makedirs(path)
else:
    print("tfrecords folder already exists.")

# Divide the data into 80% train and 20% test
train_addrs = training_list[0:int(0.8*len(training_list))]
train_labels = labels_list[0:int(0.8*len(labels_list))]

test_addrs = training_list[int(0.8*len(training_list)):]
test_labels = labels_list[int(0.8*len(labels_list)):]

def load_image(addr):
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

print("Generating the train tfrecords dict")
train_filename = 'train.tfrecords'

writer = tf.python_io.TFRecordWriter(path+train_filename)

for i in tqdm(range(len(train_addrs))):

    img = load_image(train_addrs[i])
    label = train_labels[i]
    #Create a feature
    feature = {'train/label': _int64_feature(label),
               'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    #Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    #Serialize to string and write on the file
    writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()

print("Generating the test tfrecords dict")
test_filename = 'test.tfrecords'

write = tf.python_io.TFRecordWriter(path+test_filename)

for i in tqdm(range(len(test_addrs))):

    img = load_image(test_addrs[i])
    label = test_labels[i]
    feature = {'test/label': _int64_feature(label),
               'test/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()
