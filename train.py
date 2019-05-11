import tensorflow as tf
import numpy as np
import os
import config
from tqdm import tqdm
from pathlib import Path
import readTfrecords
import gc

# Hyperparameters and other useful variables
BATCH_SIZE = config.batch_size
IMG_SIZE = config.img_size
TRAINING_EPOCHS = config.training_epochs
NUM_CLASSES = config.num_classes
LEARNING_RATE = config.learning_rate
DROPOUT = config.dropout
NUM_CHANNELS = config.num_channels

# checkpoint 
saved_model_dir = "./checkpoints"
checkpoint_file = Path("./checkpoints/checkpoint")


cwd = os.getcwd()

# Opening the tfrecords file to be used as a serialized example
def openTfRecords(phase):

    data_path = 'tfrecords/%s.tfrecords'%phase
    # Create a list of filenames and pass it to a queue
    filename = tf.train.string_input_producer([data_path])
    # Created a reader
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename)

    return serialized_example

# Getting number of iteration in the tfrecords file from number of sample
# divided by the batch size
def get_steps_per_epoch(phase):
    count = 0
    for example in tf.python_io.tf_record_iterator('tfrecords/%s.tfrecords'%phase):
        result = tf.train.Example.FromString(example)
        count +=1
    iters = int(count/BATCH_SIZE)
    print("%s_iters                                 : "%phase,  iters)
    print("Number of records found in the %sing set : "%phase, count)
    return (iters)

# Create a checkpoints folder, if it does not exist, and set the
# use_pretrained_model flag accordingly
if not (os.path.isdir("%s/checkpoints"%cwd)):
    os.mkdir("%s/checkpoints"%cwd)
    use_pretrained_model = True
else:
    use_pretrained_model = False

# Define the placeholders
def placeholder_inputs(BATCH_SIZE):
    with tf.variable_scope('inputs'):
        images_placeholder = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, NUM_CHANNELS], name="images")
        labels_placeholder = tf.placeholder(tf.int64, shape=(BATCH_SIZE), name="label")
        keep_prob_placeholder = tf.placeholder(tf.float32) # keep probability

    print('Shape of placeholders',images_placeholder.shape, labels_placeholder.shape)
    print("Training set (images) shape: {shape}".format(shape=images_placeholder.shape))
    print("Training set (labels) shape: {shape}".format(shape=labels_placeholder.shape))
    return images_placeholder, labels_placeholder, keep_prob_placeholder

# Customized conv2d layer function
def conv2d(images_placeholder, W, b, strides=1):
    images_placeholder = tf.nn.conv2d(images_placeholder, W, strides=[1, strides, strides, 1], padding='SAME')
    images_placeholder = tf.nn.bias_add(images_placeholder, b)
    return tf.nn.relu(images_placeholder)

# Maxpooling layer function
def maxpool2d(x, k=2):
    # Padding=SAME to tell tensorflow to reserve the tensor's dimensions (width and height)
    # K = pool size
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

# Network architecture
def conv_net(images_placeholder, weights, biases, dropout):
    # reshape input to 64x64 size
    images_placeholder = tf.reshape(images_placeholder, shape=[-1, IMG_SIZE, IMG_SIZE, 3])
    # Convolution layer 1
    conv1 = conv2d(images_placeholder, weights['wc1'], biases['bc1'])
    # Max pooling
    conv1 = maxpool2d(conv1, k=2)
    # Convolution layer 2
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max pooling
    conv2 = maxpool2d(conv2, k=2)
    # Convolutional layer 3
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max pooling
    conv3 = maxpool2d(conv3, k=2)
    # Convolutional layer 4
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    # Max pooling
    conv4 = maxpool2d(conv4, k=2)
    # Convolutional layer 5
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
    # Max pooling
    conv5 = maxpool2d(conv5, k=2)

    # Fully connected layer
    fc1 = tf.reshape(conv5, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, DROPOUT)

    fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, DROPOUT)


    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, wd):
  var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
  if wd is not None:
    weight_decay = tf.nn.l2_loss(var)*wd
    tf.add_to_collection('weightdecay_losses', weight_decay)
  return var

# Training function
def run_training():

    # with tf.Graph().as_default():
        tf.reset_default_graph()

        images_placeholder, labels_placeholder, keep_prob_placeholder = placeholder_inputs(
                        BATCH_SIZE
                        )
        # Output of each conv layer:
        # O = ((W-K+2P)/S)+1
        # (256-5+(2*2)/1) +1
        with tf.variable_scope('weights'):
            weights = {
                # 1st layer input 256*256 -> /2 (max pooling) will result in an
                # output of 128*128
                'wc1': _variable_with_weight_decay('wc1', [5, 5, 3, 32], 0.00005),
                'wc2': _variable_with_weight_decay('wc2', [5, 5, 32, 64], 0.00005),
                'wc3': _variable_with_weight_decay('wc3', [5, 5, 64, 128], 0.00005),
                'wc4': _variable_with_weight_decay('wc4', [5, 5, 128, 256], 0.00005),
                'wc5': _variable_with_weight_decay('wc5', [5, 5, 256, 512], 0.00005),
                # After 5 conv layers  the output is 8*8
                'wd1': _variable_with_weight_decay('wd1', [8*8*512, 1024], 0.00005),
                'wd2': _variable_with_weight_decay('wd2', [1024, 1024], 0.00005),
                'out': _variable_with_weight_decay('wout', [1024, NUM_CLASSES], 0.00005)

            }
        with tf.variable_scope('biases'):
            biases = {
                'bc1': _variable_with_weight_decay('bc1', [32], 0.000),
                'bc2': _variable_with_weight_decay('bc2', [64], 0.000),
                'bc3': _variable_with_weight_decay('bc3', [128], 0.000),
                'bc4': _variable_with_weight_decay('bc4', [256], 0.000),
                'bc5': _variable_with_weight_decay('bc5', [512], 0.000),
                'bd1': _variable_with_weight_decay('bd1', [1024], 0.000),
                'bd2': _variable_with_weight_decay('bd2', [1024], 0.000),
                'out': _variable_with_weight_decay('bout', [NUM_CLASSES], 0.000)
            }

        with tf.variable_scope("model"):
            # Creating a model
            model = conv_net(images_placeholder, weights, biases, keep_prob_placeholder)

        with tf.variable_scope("loss"):
            # Defining loss and optimizer
            cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=labels_placeholder))

        with tf.variable_scope("optimizer"):
            # Optimization method
            # now after having the cost measurement, we need to minimize it, so we create an
            # optimizer. in this case it is going to be the AdamOptimizer which is an advanced
            # form of gradient descent, which maintains the a per-parameter learning rate
            # to update the weights
            optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
            loss_sum = tf.summary.scalar('loss', cost)

        # Evaluating the model
        # This is a vector of booleans that tells whether the predicted class equals
        # the true class of each image
        correct_model = tf.equal(tf.argmax(model,1), labels_placeholder)

        # Calculating the classification accuracy by first type-casting the vector
        # of booleans to floats, so that false becomes 0 and true becomes 1, and then
        # calculating the average of these numbers
        accuracy = tf.reduce_mean(tf.cast(correct_model, tf.float32))
        accuracy_sum = tf.summary.scalar('accuracy', accuracy)

        # Create a saver to save and load the checkpoints
        saver = tf.train.Saver()

        # Launching the graph
        # Create a session for running Ops on the Graph.
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Initializing the variables inside the session
        sess.run(init)

        # If an older checkpoint exists, load the weights
        if (use_pretrained_model==False and checkpoint_file.exists()):
            ckpt = tf.train.get_checkpoint_state(saved_model_dir)
            epoch = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])+1
            print("Loading pre-trained weights from epoch ", epoch-1)
            saver.restore(sess, saved_model_dir+"/allvshem-"+str(epoch-1))
            print("Weights were successfully loaded")
            print("EPOCH_NUMBER: ", epoch)
        # Or start from the beginning
        else:
            epoch = 0
            print("No older model detected, starting a new training ")

        # Open the tfrecords files
        train_serialized_example = openTfRecords('train')
        test_serialized_example = openTfRecords('test')

        # Get the number of samples in the tfrecords
        training_iters = get_steps_per_epoch('train')
        # test_iters = get_steps_per_epoch('test')

        loss_t = []
        steps_t = []
        acc_t = []

        # Print hyper-parameters and other usefull stuff

        print("BATCH_SIZE                                  : ", BATCH_SIZE)
        print("IMG_SIZE                                    : ", IMG_SIZE)
        print("NB_of_CLASSES                               : ", NUM_CLASSES)
        print("TRAINING_EPOCHS                             : ", TRAINING_EPOCHS)
        print("DROPOUT                                     : ", DROPOUT)
        print("LEARNING_RATE                               : ", LEARNING_RATE)

        # Graph visualization
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./tensorboard/train', sess.graph)
        test_writer = tf.summary.FileWriter('./tensorboard/test', sess.graph)

        # Start the training
        iter=(epoch*training_iters)
        while epoch in range(TRAINING_EPOCHS):
            print('epoch: ', epoch)
            for step in tqdm(range(training_iters)):
                iter+=1
                train_images, train_labels = readTfrecords.extract_TfRecords(
                                serialized_example=train_serialized_example,
                                sess=sess,
                                batch_size=BATCH_SIZE
                                )
                sess.run(optimizer, feed_dict={
                                images_placeholder: train_images,
                                labels_placeholder: train_labels,
                                keep_prob_placeholder: 0.8
                                })

                if (iter%training_iters == 0):
                    print('*' * 15)
                    summary, loss, acc = sess.run([merged, cost, accuracy], feed_dict={
                                images_placeholder: train_images,
                                labels_placeholder: train_labels,
                                keep_prob_placeholder: 1.0
                                })
                    print("Iter " + str(iter) + ", Loss = " + \
                          "{:.3f}".format(loss) + ", Training Accuracy = " + \
                          "{:.3f}".format(acc))

                    # Can be used for plotting
                    loss_t.append(loss)
                    steps_t.append(step*BATCH_SIZE)
                    acc_t.append(acc)

                    train_writer.add_summary(summary, global_step=epoch)
                    print('Saving checkpoint', epoch)
                    save_path = saver.save(sess, os.path.join(saved_model_dir, 'allvshem'), global_step=epoch)
                    test_images, test_labels = readTfrecords.extract_TfRecords(
                                        serialized_example=test_serialized_example,
                                        sess=sess,
                                        batch_size=BATCH_SIZE
                                        )
                    summary, loss, acc = sess.run([merged, cost, accuracy], feed_dict={
                                        images_placeholder: test_images,
                                        labels_placeholder: test_labels,
                                        keep_prob_placeholder: 1.0
                                        # learning_rate_placeholder: learning_rate
                                        })
                    print('Testing loss: ' + '{:.3f}'.format(loss) + ', Testing accuracy: ' + '{:.3f}'.format(acc) )
                    test_writer.add_summary(summary, global_step=epoch)
                    epoch+=1
        print('done')

def main(_):
  run_training()

if __name__ == '__main__':
  tf.app.run()
