import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import config
import sys

# batch_size = config.batch_size
# num_epochs = 60
img_size = config.img_size
# phase = 'train'
def extract_TfRecords(serialized_example, sess, batch_size):

    feature = dict()
    feature['label']=tf.FixedLenFeature([], tf.int64)
    feature['image']=tf.FixedLenFeature([], tf.string)

    parsed_features = tf.parse_single_example(serialized_example, feature)

    # Convert the image data from string back to the numbers
    image = tf.decode_raw(parsed_features['image'.format()], tf.float32)

    # Cast label data into int64
    label = tf.cast(parsed_features['label'], tf.int64)

    # Reshape image data into the original shape
    image = tf.reshape(image, [config.img_size, config.img_size, 3])

    # Creates batches by randomly shuffling tensors
    min_after_dequeue = 50
    capacity = 500
    images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                            num_threads=1, min_after_dequeue=min_after_dequeue)

    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)



    while True:
        try:
            # for batch_index in range(5):
            train_images, train_labels = sess.run([images, labels])
            train_images = train_images.astype(np.int64)
            # print(train_labels)
                    # print(train_images)
        # for j in range(6):
        #     plt.subplot(2, 3, j+1)
        #     plt.imshow(train_images[j, ...])
        #     plt.title('hem' if train_labels[j]==0 else 'all')
        # plt.show()
            return train_images, train_labels
        except tf.errors.OutOfRangeError:
                print("End of dataset")

    # # Stop the threads
        coord.request_stop()
            # sess.run(coord.queue.close(cancel_pending_enqueues=True))
        coord.join(threads)
    sess.close()
