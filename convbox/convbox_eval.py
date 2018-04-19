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
#from utils.wrapper import nms
from utils_ov.cython_nms import nms
from copy import copy
#from utils.wrapper import tensor_to_box

FLAGS = gflags.FLAGS
# Basic model parametersh
gflags.DEFINE_enum('dataset', 'pascal_voc', ['pascal_voc', 'coco', 'kitti'], 
                            """Name of dataset.""")
gflags.DEFINE_string('eval_file', '/home/ckp/data/PASCAL/testdata_lmdb',
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

                          
                          
                          
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
 "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
 "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
''' 
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
'''
def apply_nms(boxes, scores):

    CONF_THRESH = 0.001
    NMS_THRESH = 0.5
    
    
    res = {}
    for cls_ind in xrange(20):
        cls_boxes = copy(boxes)
        cls_scores = scores[:, cls_ind]
        keep = np.where(cls_scores >= CONF_THRESH)[0]
        cls_boxes = cls_boxes[keep, :]
        cls_scores = cls_scores[keep]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        res[cls_ind] = dets

    return res

def save_result(im, imagename, boxes):
    im_id = imagename
    for key, value in boxes.items():
        filename = 'comp4_det_test_' + classes[key] + '.txt' 
        #fp = open('../pascalEval/result/' + filename, 'a')
        fp = open('/home/ckp/convbox2/result/data/' + filename, 'a+')

        for box in value:
            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[3]
            
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > im[0] - 1:
                x2 = im[0] - 1
            if y2 > im[1] - 1:
                y2 = im[1] - 1

            fp.write(str(im_id) + ' ' + str(box[4]) + ' ')
            fp.write(str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2))
            fp.write('\n')

        fp.close()
    
def evaluate():
    with tf.Graph().as_default():
      with tf.device("/cpu:0"):
        images = tf.placeholder(tf.float32, shape=(1, FLAGS.input_row, FLAGS.input_col, 3))
        #channels = FLAGS.num_box*(3+FLAGS.output_row+FLAGS.output_col+FLAGS.num_class)*2
        channels = 50
        dets = inference(images, channels)

        # restore the moving average version of the learned variables.
        variable_averages = tf.train.ExponentialMovingAverage(0.9999)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # config gpu.
        #gpu_options = tf.GPUOptions(
        #        visible_device_list=FLAGS.gpu_list, 
        #        per_process_gpu_memory_fraction=FLAGS.gpu_usage)
        sess = tf.Session(config=tf.ConfigProto(
                ))
        
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
        n=14
        for datum in ds.eval_datum_iterator():
            count += 1; print count, datum[0]
            #det_list = sess.run(dets, feed_dict={images: datum[1][0]})
            out = sess.run(dets, feed_dict={images: datum[1][0]})
            #print (out.shape)
            boxes = np.zeros(( 2 * 14 * 14, 4))
            scores = np.zeros((2 * 14 * 14, 20))
            for i in range(14):
                for j in range(14):
                    for num in range(2):
                        #ind = n * n * 2 + 2 * (i * n + j) + num
                        ind = 2 * (i * n + j) + num
                        for k in range(20):
                            scores[ind, k] = out[0, i, j, num] * out[0, i, j, 2 + k + 20*num]

                        #x = (out[s][42 + 4*num, i, j] + j) / n * im.size[0]
                        x=(out[0,i,j,42+4*num]+j)/n*448
                        y=(out[0,i,j,43+4*num]+i)/n*448
                        #y = (out[s][43 + 4*num, i, j] + i) / n * im.size[1]
                        #w = (out[s][44 + 4*num, i, j] ** 2) * im.size[0]
                        w=(out[0,i,j,44+4*num]**2)*448
                        h=(out[0,i,j,45+4*num]**2)*448
                        #h = (out[s][45 + 4*num, i, j] ** 2) * im.size[1]

                        boxes[ind, 0] = x - w/2
                        boxes[ind, 1] = y - h/2
                        boxes[ind, 2] = x + w/2
                        boxes[ind, 3] = y + h/2
            res  = apply_nms(boxes,scores)
            #boxes = tensor_to_box(det_list, FLAGS.num_box, datum[1][1], pool)
            #res = apply_nms(boxes, 0.55, int(FLAGS.gpu_list))
            save_result(datum[1][1], datum[0], res)
            #ds.save_result(datum[0], datum[1][1], res, FLAGS.eval_dir)
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
