# --------------------------------------------------------
# ConvBox
# Copyright (c) 2016 HUST
# Licensed under The MIT License [see LICENSE for details]
# Written by Ckb
# --------------------------------------------------------
from __future__ import division

import sys
import cv2
import time
import gflags
import numpy as np
from datetime import datetime
import multiprocessing

import tensorflow as tf
from convbox_model import inference
from dataset.interface import dataset
from utils.wrapper import nms
from utils.wrapper import tensor_to_box

FLAGS = gflags.FLAGS
# Basic model parametersh
gflags.DEFINE_enum('dataset', 'pascal_voc', ['pascal_voc', 'coco', 'kitti'], 
                            """Name of dataset.""")
gflags.DEFINE_string('eval_file', 'data/pascal_voc_test_2007_lmdb',
                            """Path to the eval file.""")
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
gflags.DEFINE_string('eval_dir', 'result',
                           """Directory where to write result files..""")
gflags.DEFINE_string('checkpoint_path', 'backup/model.ckpt-50000',
                           """Model file after training.""")
gflags.DEFINE_string('gpu_list', '1',
                           """GPU list, such as '0, 1, ....'.""")
gflags.DEFINE_float('gpu_usage', 0.8,
                          """Per process gpu memory fraction.""")

    
def apply_nms(boxes, nms_thresh, gpu_id):

    CONF_THRESH = 0.001
  
    res = []
    for cls_ind in xrange(FLAGS.num_class):
        if boxes[cls_ind].size == 0:
            res.append(np.zeros((0, 5), dtype=np.float32))
            continue
        cls_scores = boxes[cls_ind][:, 4]
        keep = np.where(cls_scores >= CONF_THRESH)[0]
        dets = boxes[cls_ind][keep, :]
        keep = nms(dets, nms_thresh, gpu_id)
        res.append(dets[keep, :])
                
    return res

    
def evaluate():
    with tf.Graph().as_default():
        # build graph.
        images = tf.placeholder(tf.float32, shape=(1, FLAGS.input_row, FLAGS.input_col, 3))
        #channels = FLAGS.num_box*(3+FLAGS.output_row+FLAGS.output_col+FLAGS.num_class)*2
        channels = 50
        dets = inference(images, channels)

        # restore the moving average version of the learned variables.
        variable_averages = tf.train.ExponentialMovingAverage(0.9999)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # config gpu.
        gpu_options = tf.GPUOptions(
                visible_device_list=FLAGS.gpu_list, 
                per_process_gpu_memory_fraction=FLAGS.gpu_usage)
        sess = tf.Session(config=tf.ConfigProto(
                gpu_options=gpu_options))
        
        # load model.
        assert tf.gfile.Exists(FLAGS.checkpoint_path + '.index')
        saver.restore(sess, FLAGS.checkpoint_path)
        print '%s: Model restored from %s.' %(
                datetime.now(), FLAGS.checkpoint_path)
        
        params = {'name': FLAGS.dataset,
                  'lmdb_file': FLAGS.eval_file, 
                  'input_shape': [FLAGS.input_row, FLAGS.input_col], 
                  'output_shape': [FLAGS.output_row, FLAGS.output_col], 
                  'num_class': FLAGS.num_class}
        ds = dataset(params)
        print '%s: Starting evaluation on %s.' %(
                datetime.now(), FLAGS.dataset)
                
        pool = multiprocessing.Pool(processes=8)
                
        count = -1
        start_time = time.time()
        for datum in ds.eval_datum_iterator():
            count += 1; print count, datum[0]
            det_list = sess.run(dets, feed_dict={images: datum[1][0]})
            boxes = tensor_to_box(det_list, FLAGS.num_box, datum[1][1], pool)
            res = apply_nms(boxes, 0.55, int(FLAGS.gpu_list))
            ds.save_result(datum[0], datum[1][1], res, FLAGS.eval_dir)
        duration = time.time() - start_time
        print '%s: Use time: %d seconds.' %(datetime.now(), duration)
        
        pool.close()
        pool.join()
        
        
def main(unused_argv=None):
    FLAGS(sys.argv)
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()

if __name__ == '__main__':
    tf.app.run()
