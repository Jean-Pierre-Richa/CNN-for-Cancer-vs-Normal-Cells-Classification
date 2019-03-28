import tensorflow as tf
import numpy as np
import os
import config
from tqdm import tqdm
from pathlib import Path
import readTfrecords
import gc

DATASET_DIR = config.DATASET_DIR
BATCH_SIZE = config.batch_size
img_size = config.img_size
TRAINING_EPOCHS = config.training_epochs
NUM_CLASSES = config.NUM_CLASSES
LEARNING_RATE = config.learning_rate

saved_model_dir = "./checkpoints"
checkpointFile = Path("./checkpoints/checkpoint")
cwd = os.getcwd()

def openTfRecords(phase):

    data_path = 'tfrecords/%s.tfrecords'%phase
    # Create a list of filenames and pass it to a queue
    filename = tf.train.string_input_producer([data_path])
    # Created a reader
    reader = tf.TFRecordReader()
    #
    _, serialized_example = reader.read(filename)

    return serialized_example

def get_steps_per_epoch(phase):
    count = 0
    for example in tf.python_io.tf_record_iterator('tfrecords/%s.tfrecords'%phase):
        result = tf.train.Example.FromString(example)
        count +=1
    iters = int(count/BATCH_SIZE)
    print("%s_iters                                 : "%phase,  iters)
    print("Number of records found in the %sing set : "%phase, count)
    return (iters)

if not (os.path.isdir("%s/checkpoints"%cwd)):
    os.mkdir("%s/checkpoints"%cwd)
    use_pretrained_model = True
else:
    use_pretrained_model = False

def placeholder_inputs(BATCH_SIZE):

    images_placeholder = tf.placeholder(tf.float32, [None, config.img_size, config.img_size, 3], name="images")
    labels_placeholder = tf.placeholder(tf.int64, shape=(BATCH_SIZE), name="label")
    keep_prob = tf.placeholder(tf.float32) # keep probability

    print('Shape of placeholders',images_placeholder.shape, labels_placeholder.shape)
    print("Training set (images) shape: {shape}".format(shape=images_placeholder.shape))
    print("Training set (labels) shape: {shape}".format(shape=labels_placeholder.shape))
    return images_placeholder, labels_placeholder, keep_prob

def conv2d(images_placeholder, W, b, strides=1):
    images_placeholder = tf.nn.conv2d(images_placeholder, W, strides=[1, strides, strides, 1], padding='SAME')
    images_placeholder = tf.nn.bias_add(images_placeholder, b)
    return tf.nn.relu(images_placeholder)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def conv_net(images_placeholder, weights, biases, dropout):
    # reshape input to 64x64 size
    images_placeholder = tf.reshape(images_placeholder, shape=[-1, img_size, img_size, 3])
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

    # Fully connected layer
    fc1 = tf.reshape(conv4, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, config.dropout)

    fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, config.dropout)


    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out

    # [5, 5, 3, 32]: 5, 5 are the filter's height and width
                   # 3, 32 are the number of input and output channels respectively
                   # grayscale = 1 RGBD = 3, here grayscale
# with tf.name_scope(''):
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

def run_training():

    # with tf.Graph().as_default():
        tf.reset_default_graph()

        images_placeholder, labels_placeholder, keep_prob = placeholder_inputs(
                        BATCH_SIZE
                        )
        # (64-5+(2*2)/1) +1
        with tf.variable_scope('var_name') as var_scope:
            weights = {
                'wc1': _variable_with_weight_decay('wc1', [5, 5, 3, 64], 0.0005),
                'wc2': _variable_with_weight_decay('wc2', [5, 5, 64, 128], 0.0005),
                'wc3': _variable_with_weight_decay('wc3', [5, 5, 128, 256], 0.0005),
                'wc4': _variable_with_weight_decay('wc4', [5, 5, 256, 512], 0.0005),
                # 'wc5': _variable_with_weight_decay('wc5', [5, 5, 512, 1024], 0.0005),
                'wd1': _variable_with_weight_decay('wd1', [4*4*512, 2048], 0.0005),
                'wd2': _variable_with_weight_decay('wd2', [2048, 2048], 0.0005),
                'out': _variable_with_weight_decay('wout', [2048, config.NUM_CLASSES], 0.0005)

                # 'wc1': tf.Variable(tf.random_normal([5, 5, 3, 64]), name='wc1'),
                # 'wc2': tf.Variable(tf.random_normal([5, 5, 64, 128]), name='wc2'),
                # 'wc3': tf.Variable(tf.random_normal([5, 5, 128, 256]), name='wc3'),
                # 'wc4': tf.Variable(tf.random_normal([5, 5, 256, 512]), name='wc4'),
                # 'wd1': tf.Variable(tf.random_normal([4*4*512, 2048]), name='wd1'),
                # 'wd2': tf.Variable(tf.random_normal([2048, 2048]), name='wd2'),
                # 'out': tf.Variable(tf.random_normal([2048, config.NUM_CLASSES]), name='wout')
            }

            biases = {
                'bc1': _variable_with_weight_decay('bc1', [64], 0.000),
                'bc2': _variable_with_weight_decay('bc2', [128], 0.000),
                'bc3': _variable_with_weight_decay('bc3', [256], 0.000),
                'bc4': _variable_with_weight_decay('bc4', [512], 0.000),
                # 'bc5': _variable_with_weight_decay('bc5', [1024], 0.000),
                'bd1': _variable_with_weight_decay('bd1', [2048], 0.000),
                'bd2': _variable_with_weight_decay('bd2', [2048], 0.000),
                'out': _variable_with_weight_decay('bout', [config.NUM_CLASSES], 0.000)

                # 'bc1': tf.Variable(tf.random_normal([64]), name='bc1'),
                # 'bc2': tf.Variable(tf.random_normal([128]), name='bc2'),
                # 'bc3': tf.Variable(tf.random_normal([256]), name='bc3'),
                # 'bc4': tf.Variable(tf.random_normal([512]), name='bc4'),
                # 'bd1': tf.Variable(tf.random_normal([2048]), name='bd1'),
                # 'bd2': tf.Variable(tf.random_normal([2048]), name='bd2'),
                # 'out': tf.Variable(tf.random_normal([config.NUM_CLASSES]), name='bout')
            }

            # restore_variables = [v.name for v in tf.trainable_variables(scope='var_name')]
            # variables_to_restore = tf.contrib.framework.get_variables_to_restore(include=restore_variables)

            # Creating a model
            model = conv_net(images_placeholder, weights, biases, keep_prob)

            # Defining loss and optimizer
            cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=labels_placeholder))

            # Optimization method
            # now after having the cost measurement, we need to minimize it, so we create an
            # optimizer. in this case it is going to be the AdamOptimizer which is an advanced
            # form of gradient descent, which maintains the a per-parameter learning rate
            # to update the weights
            optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(cost)
            loss_sum = tf.summary.scalar('loss', cost)

            # Evaluating the model
            # y_true_cls = tf.argmax(labels_placeholder, 1)
            y_true_cls = labels_placeholder
            y_pred_cls = tf.argmax(model, 1)

            # This is a vector of booleans that tells whether the predicted class equals
            # the true class of each image
            correct_model = tf.equal(y_pred_cls, y_true_cls)

            # Calculating the classification accuracy by first type-casting the vector
            # of booleans to floats, so that false becomes 0 and true becomes 1, and then
            # calculating the average of these numbers
            accuracy = tf.reduce_mean(tf.cast(correct_model, tf.float32))
            accuracy_sum = tf.summary.scalar('accuracy', accuracy)

            # tf.add_to_collection('vars', images_placeholder)
            # tf.add_to_collection('vars', labels_placeholder)
            # saver_variables = tf.trainable_variables(scope='var_name')
            # saver = tf.train.Saver(saver_variables)
            saver = tf.train.Saver()

            # Launching the graph
            # Create a session for running Ops on the Graph.
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

            # initializing the variables
            init = tf.global_variables_initializer()

            sess.run(init)

            if (use_pretrained_model==False and checkpointFile.exists()):
                ckpt = tf.train.get_checkpoint_state(saved_model_dir)
                epoch = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])+1
                print("Loading pre-trained weights from epoch ", epoch-1)
                saver.restore(sess, saved_model_dir+"/allvshem-"+str(epoch-1))
                print("Weights were successfully loaded")
                print("epoch number: ", epoch)
            else:
                epoch = 0
                print("No older model detected, starting a new training ")

            train_serialized_example = openTfRecords('train')
            test_serialized_example = openTfRecords('test')

            training_iters = get_steps_per_epoch('train')
            # test_iters = get_steps_per_epoch('test')

            loss_t = []
            steps_t = []
            acc_t = []

            print("Batch_size                                  : ", BATCH_SIZE)
            print("Learning Rate                               : ", LEARNING_RATE)
            print("Number of classes to predict                : ", NUM_CLASSES)
            print("Training epochs                             : ", TRAINING_EPOCHS)


            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('./tensorboard/train', sess.graph)
            test_writer = tf.summary.FileWriter('./tensorboard/test', sess.graph)
            # print(steps_per_epoch)
            iter=(epoch*training_iters)
            while epoch in range(config.training_epochs):
                # learning_rate = config.learning_rate-epoch/1000
                print('epoch: ', epoch)
                for step in tqdm(range(training_iters)):
                    iter+=1
                # for step in tqdm(range(1)):
                    train_images, train_labels = readTfrecords.extract_TfRecords(
                                    serialized_example=train_serialized_example,
                                    sess=sess,
                                    batch_size=BATCH_SIZE
                                    )
                    # print('batch_size: ', batch_size)
                    sess.run(optimizer, feed_dict={
                                    images_placeholder: train_images,
                                    labels_placeholder: train_labels,
                                    keep_prob: config.dropout
                                    })
                    # print("step ", step)
                    if ((iter)%training_iters == 0):
                        # print("step ", step)
                        print('*' * 15)
                        summary, loss, acc = sess.run([merged, cost, accuracy], feed_dict={
                                    images_placeholder: train_images,
                                    labels_placeholder: train_labels,
                                    keep_prob: 1.
                                    })
                        print("Iter " + str(iter) + ", Loss = " + \
                              "{:.3f}".format(loss) + ", Training Accuracy = " + \
                              "{:.3f}".format(acc))

                        loss_t.append(loss)
                        steps_t.append(step*BATCH_SIZE)
                        acc_t.append(acc)
                        train_writer.add_summary(summary, global_step=epoch)
                        print('Saving checkpoint', epoch)
                        # saver.save(sess, os.path.join(saved_model_dir, 'allvshem'), global_step=epoch)
                        save_path = saver.save(sess, os.path.join(saved_model_dir, 'allvshem'), global_step=epoch)
                        # for step in tqdm(range(test_iters)):
                        test_images, test_labels = readTfrecords.extract_TfRecords(
                                            serialized_example=test_serialized_example,
                                            sess=sess,
                                            batch_size=BATCH_SIZE
                                            )
                        # print("Testing Accuracy: ", \
                        summary, loss, acc = sess.run([merged, cost, accuracy], feed_dict={
                                            images_placeholder: test_images,
                                            labels_placeholder: test_labels,
                                            keep_prob: 1.
                                            })
                        print('Testing accuracy: ' + '{:.3f}'.format(acc))
                        test_writer.add_summary(summary, global_step=epoch)
                            # sess.close()
                        epoch+=1
                        # gc.collect()

            print('done')

def main(_):
  run_training()

if __name__ == '__main__':
  tf.app.run()
