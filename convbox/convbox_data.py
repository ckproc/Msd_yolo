# --------------------------------------------------------
# ConvBox
# Copyright (c) 2016 HUST
# Licensed under The MIT License [see LICENSE for details]
# Written by Ckb
# --------------------------------------------------------

import gflags
import threading
import numpy as np

import tensorflow as tf
from dataset.interface import dataset


FLAGS = gflags.FLAGS
# Basic model parameters.
gflags.DEFINE_enum('dataset', 'pascal_voc', ['pascal_voc', 'coco', 'kitti'], 
                            """Name of dataset.""")
gflags.DEFINE_string('train_file', '/home/ckp/data/PASCAL/VOCdevkit/VOC0712/train_val_lmdb',
                            """Path to the lmdb file.""")
gflags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
gflags.DEFINE_integer('input_row', 448,
                            """Provide row of input image.""")
gflags.DEFINE_integer('input_col', 448,
                            """Provide col of input image.""")
gflags.DEFINE_integer('output_row', 14,
                            """Provide row of output shape.""")
gflags.DEFINE_integer('output_col', 14,
                            """Provide col of output shape.""")
gflags.DEFINE_integer('num_class', 20,
                            """Number of class of dataset.""")


class CustomRunner(object):
    """
    This class manages the the background threads needed to fill
    a queue full of data.
    """
    
    def __init__(self):
        params = {'name': FLAGS.dataset,
                  'lmdb_file': FLAGS.train_file, 
                  'input_shape': [FLAGS.input_row, FLAGS.input_col], 
                  'output_shape': [FLAGS.output_row, FLAGS.output_col], 
                  'num_class': FLAGS.num_class}
        self.dataset = dataset(params)

        channels = (3+FLAGS.output_row+FLAGS.output_col+FLAGS.num_class)*2
        shapes = [[FLAGS.input_row, FLAGS.input_col, 3],
                  [FLAGS.output_row, FLAGS.output_col, channels]]
        self.image = tf.placeholder(dtype=tf.float32, shape=shapes[0])
        self.label = [tf.placeholder(dtype=tf.float32, shape=shapes[1]) for _ in xrange(4)]
        # The actual queue of data.
        self.queue = tf.FIFOQueue(shapes=[shapes[0]] + [shapes[1]] * 4, 
                                  dtypes=[tf.float32] * 5,
                                  capacity=640)
        tf.summary.scalar('queue_size', self.queue.size())

        # The symbolic operation to add data to the queue
        self.enqueue_op = self.queue.enqueue([self.image] + self.label)

    def get_inputs(self):
        """
        Return's tensors containing a batch of images and labels
        """
        batch = self.queue.dequeue_many(FLAGS.batch_size)
        images_batch, labels_batch = batch[0], batch[1:]
        tf.summary.image('images_batch', images_batch)
        return images_batch, labels_batch

    def thread_func(self, sess):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        for datum in self.dataset.train_datum_iterator:
            feed_dict = {self.image: datum[0]}
            for i in xrange(4):
                feed_dict[self.label[i]] = datum[1][i]
            sess.run(self.enqueue_op, feed_dict=feed_dict)

    def start_threads(self, sess, n_threads=1):
        """ Start background threads to feed queue """
        threads = []
        for n in xrange(n_threads):
            t = threading.Thread(target=self.thread_func, args=(sess,))
            t.daemon = True # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads


class SimpleRunner(object):
    def __init__(self):
        params = {'name': FLAGS.dataset,
                  'lmdb_file': FLAGS.train_file, 
                  'input_shape': [FLAGS.input_row, FLAGS.input_col], 
                  'output_shape': [FLAGS.output_row, FLAGS.output_col], 
                  'num_class': FLAGS.num_class}
        self.dataset = dataset(params)

        #channels = (3+FLAGS.output_row+FLAGS.output_col+FLAGS.num_class)*2
        channels = 25
        #shapes = [[FLAGS.batch_size, FLAGS.input_row, FLAGS.input_col, 3],
        #          [FLAGS.batch_size, FLAGS.output_row, FLAGS.output_col, channels]]
        
        shapes = [[FLAGS.batch_size, FLAGS.input_row, FLAGS.input_col, 3],
                  [FLAGS.batch_size, FLAGS.output_row, FLAGS.output_col, channels]]
        self.images = tf.placeholder(dtype=tf.float32, shape=shapes[0])
        self.labels = tf.placeholder(dtype=tf.float32, shape=shapes[1]) 

    def get_inputs(self):
        return self.images, self.labels

    def get_next_batch_datums(self):
        #batch_images, batch_labels = [], [[] for _ in xrange(4)]
        batch_images, batch_labels = [], []
        for _ in xrange(FLAGS.batch_size):
            datum = self.dataset.next_train_datum
            batch_images.append(datum[0])
            batch_labels.append(datum[1])
            #for i in xrange(4):
            #    batch_labels[i].append(datum[1][i])
        #return np.asarray(batch_images), [np.asarray(batch_labels[i]) for i in xrange(4)]
        return np.asarray(batch_images), np.asarray(batch_labels)
