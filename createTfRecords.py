from random import shuffle
import glob
import os, sys
import config
import tensorflow as tf
import cv2
import numpy as np
from tqdm import tqdm
import augmentSet
import random

DATASET = config.DATASET_DIR
img_size = config.img_size
all_list = list()
training_list = list()
labels_list = list()
newDS = 'fold_3'

for foldNum in os.listdir(DATASET):
    for category in os.listdir(os.path.join(DATASET, foldNum)):
        for images in os.listdir(os.path.join(DATASET, foldNum, category)):
            all_list.append(os.path.join(DATASET,foldNum,category,images))
random.shuffle(all_list)

for i in range(len(all_list)):
    name = all_list[i].split('/')[5]
    # print(name)
    if name.startswith('rotated'):
        nameSplit = name.split('_')[5]
    else:
        # print(name)
        nameSplit = name.split('_')[4]
    categ = nameSplit.split('.')[0]

    if(categ == 'all'):
        training_list.append(all_list[i])
        labels_list.append(1)
    else:
        training_list.append(all_list[i])
        labels_list.append(0)

path = './tfrecords/'
if not os.path.isdir('./tfrecords'):
    print("creating tfrecords folder")
    os.makedirs(path)
else:
    print("tfrecords folder already exists.")
if not os.path.isdir(os.path.join(DATASET, newDS)):
    print('Augmenting the dataset')
    training_list, labels_list = augmentSet.augment(training_list, labels_list)
else:
    print('Augmented dataset folder detected')

# Divide the data into 80% train and 20% test
train_addrs = training_list[0:int(0.8*len(training_list))]
train_labels = labels_list[0:int(0.8*len(labels_list))]
# print ('train_addrs ', train_addrs)
test_addrs = training_list[int(0.8*len(training_list)):]
test_labels = labels_list[int(0.8*len(labels_list)):]
# print ('test_addrs ', test_addrs)
def load_image(addr):
    img = cv2.imread(addr)
    img = cv2.resize(img, dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img

def _int64_feature(value):
    # if not isinstance(value, list):
        # value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

print("Generating the train tfrecords dict")
train_filename = 'train.tfrecords'

writer_train = tf.python_io.TFRecordWriter(path+train_filename)

for i in tqdm(range(len(train_addrs))):

    img_train = load_image(train_addrs[i])
    label_train = train_labels[i]

    # print('label', label)
    #Create a feature
    feature = {}
    feature ['label'] =  _int64_feature(label_train)
    feature ['image'] =  _bytes_feature(tf.compat.as_bytes(img_train.tobytes()))
    # print(label)
    #Create an example protocol buffer
    train_example = tf.train.Example(features=tf.train.Features(feature=feature))
    #Serialize to string and write on the file
    writer_train.write(train_example.SerializeToString())

writer_train.close()
sys.stdout.flush()

print("Generating the test tfrecords dict")
test_filename = 'test.tfrecords'

writer_test = tf.python_io.TFRecordWriter(path+test_filename)

for j in tqdm(range(len(test_addrs))):

    img_test = load_image(test_addrs[j])
    label_test = test_labels[j]
    feature = {}
    feature ['label'] =  _int64_feature(label_test)
    feature ['image'] =  _bytes_feature(tf.compat.as_bytes(img_test.tobytes()))
    test_example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer_test.write(test_example.SerializeToString())

writer_test.close()
sys.stdout.flush()
