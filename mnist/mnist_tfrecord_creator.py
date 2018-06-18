import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data


def main():
    
    mnist = input_data.read_data_sets("data/", one_hot=True)
    tick = int(time.time())
    train_file = 'mnist-{}-train.tfrecords'.format(tick)
    test_file = 'mnist-{}-test.tfrecords'.format(tick)
    test_writer = tf.python_io.TFRecordWriter(test_file)
    train_writer = tf.python_io.TFRecordWriter(train_file)
    data = mnist.train.next_batch(mnist.train.num_examples)

    for i, img in enumerate(data[0]):
        features = {
            'image': tf.train.Feature(float_list=tf.train.FloatList(value=img.flatten().tolist())),
            'label': tf.train.Feature(float_list=tf.train.FloatList(value=data[1][i].flatten().tolist())),
            'id': tf.train.Feature(float_list=tf.train.FloatList(value=[i])),
        }
        train_writer.write(tf.train.Example(
        features=tf.train.Features(feature=features)).SerializeToString())

    data = mnist.test.next_batch(mnist.test.num_examples)

    for i, img in enumerate(data[0]):
        features = {
            'image': tf.train.Feature(float_list=tf.train.FloatList(value=img.flatten().tolist())),
            'label': tf.train.Feature(float_list=tf.train.FloatList(value=data[1][i].flatten().tolist())),
            'id': tf.train.Feature(float_list=tf.train.FloatList(value=[i])),
        }
        test_writer.write(tf.train.Example(
        features=tf.train.Features(feature=features)).SerializeToString())
    
if __name__ == '__main__':
    main()
