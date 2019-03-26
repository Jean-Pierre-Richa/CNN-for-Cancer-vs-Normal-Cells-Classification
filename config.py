# Config file that contains common variables

# dataset directory
DATASET_DIR = "./dataset/train/"
tfRecords_train = "./tfrecords/train.tfrecords"
tfRecords_test  = "./tfrecords/test.tfrecords"
#Number of classes, one class for each of the 10 digits
NUM_CLASSES = 2
# image size
img_size = 64
# Number of color channels for the images: 1 channel for gray-scale
num_channels = 3
# Images are stored in one-dimensional arrays of the length
img_size_flat = img_size*img_size
# Tuple with height and width of images used to reshape arrays
img_shape = (img_size, img_size)

# Architecture hyper parameters
# learning rate
learning_rate = 0.001
# number of epochs
training_epochs = 60
# batch size
batch_size = 32
# 64x64 image
n_input = img_size_flat
# Dropout
dropout = 0.4
