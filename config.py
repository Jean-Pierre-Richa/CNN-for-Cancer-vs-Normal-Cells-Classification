# Config file that contains common variables

# dataset directory
XDATASET_DIR = "dataset/train/"
# YDATASET_DIR = "converted-DS/labels.npy"

#Number of classes, one class for each of the 10 digits
NUM_CLASSES = 10
# image size
img_size = 64
# Number of color channels for the images: 1 channel for gray-scale
num_channels = 1
# Images are stored in one-dimensional arrays of the length
img_size_flat = img_size*img_size
# Tuple with height and width of images used to reshape arrays
img_shape = (img_size, img_size)

# Architecture hyper parameters
# learning rate
learning_rate = 0.005
# number of iterations
training_iters = 100000
# batch size
batch_size = 20

display_step = 10
# 64x64 image
n_input = img_size_flat

dropout = 0.75
