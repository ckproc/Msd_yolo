# --------------------------------------------------------
# ConvBox
# Copyright (c) 2016 HUST
# Licensed under The MIT License [see LICENSE for details]
# Written by Ckb
# --------------------------------------------------------
from __future__ import division

import os
import sys
import cv2
import time
import gflags
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt

import tensorflow as tf
from convbox_model import inference
from dataset.interface import dataset
from utils.wrapper import nms


FLAGS = gflags.FLAGS
# Basic model parametersh
gflags.DEFINE_enum('dataset', 'pascal_voc', ['pascal_voc', 'coco', 'kitti'], 
                            """Name of dataset.""")
gflags.DEFINE_string('eval_file', 'data/pascal_voc_test_2007_lmdb',
                            """Path to the lmdb file.""")
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


CLASSES = []
                          
COLORS = [(0, 255, 0), (0, 0, 255), (241, 90, 36), 
(235, 0, 139), (0, 159, 255), (223, 255, 0), 
(237, 34, 42), (180, 58, 228), (247, 147, 30), (2, 254, 207)]

COUNT = -1


def save_result(im_id, im_sz, boxes_list, eval_dir):

    global COUNT
    COUNT += 1

    SHOW_GT = False
    SHOW_FOUR_BRANCHES = False

    im = (im_sz[0][0] + 1) * 255 / 2.0
    im = np.uint8(im)
    im = cv2.resize(im, im_sz[1])
    
    for ind, boxes in enumerate(boxes_list):
        if (ind < 4 and not SHOW_FOUR_BRANCHES):
            continue
        if (ind == 5 and not SHOW_GT):
            continue
        fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
        ax.imshow(im.copy(), aspect='equal')
        for k, value in enumerate(boxes):
            for box in value:
                [x1, y1, x2, y2] = [int(x) for x in box[:-1]]
                
                if x1 < 0: x1 = 0
                if y1 < 0: y1 = 0
                if x2 > im_sz[1][0] - 1: x2 = im_sz[1][0] - 1
                if y2 > im_sz[1][1] - 1: y2 = im_sz[1][1] - 1
                
                ax.add_patch(plt.Rectangle((x1, y1), 
                             x2 - x1, y2 - y1, 
                             fill=False, 
                             edgecolor=[x/255.0 for x in COLORS[k%10]], 
                             linewidth=8) )
                ax.text(x1, y1 - 2, 
                        '{:s}: {:.3f}'.format(CLASSES[k], box[4]),
                        bbox=dict(facecolor='white', alpha=0.5), 
                        fontsize=20, 
                        color='black')
       
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('{}/{:0>8d}_{}.jpg'.format(
                eval_dir, COUNT, ind), bbox_inches='tight')
        plt.close()
        
        
def apply_nms(boxes, nms_thresh, gpu_id):

    CONF_THRESH = 0.1
  
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

    
def get_range(i, rows, ss):
    if i < rows/2:
        if ss < 0:
            return 0, i + 1
        else:
            return i, 2*(i + 1)
    else:
        if ss < 0:
            return 2*i - rows, i + 1
        else:
            return i, rows
    

def evaluate():
    with tf.Graph().as_default():
        # build graph.
        images = tf.placeholder(tf.float32, shape=(1, FLAGS.input_row, FLAGS.input_col, 3))
        channels = FLAGS.num_box*(3+FLAGS.output_row+FLAGS.output_col+FLAGS.num_class)*2
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
        global CLASSES
        CLASSES = ds.class_list
        print '%s: Starting evaluation on %s.' %(
                datetime.now(), FLAGS.dataset)
        count = -1
        start_time = time.time()
        for datum in ds.eval_datum_iterator(0, 300):
            count += 1; print count, datum[0]
            det_list = sess.run(dets, feed_dict={images: datum[1][0]})
            
            res_list = []
            ss = [(-1, -1), (+1, -1), (-1, +1), (+1, +1)]
            boxes = [[[] for _ in xrange(FLAGS.num_class)] for _ in xrange(4)]
            
            for k in xrange(4):
                det = det_list[k][0]

                for i in xrange(FLAGS.output_row):
                    for j in xrange(FLAGS.output_col):
                        p_start, p_end = get_range(i, FLAGS.output_row, ss[k][1])
                        q_start, q_end = get_range(j, FLAGS.output_col, ss[k][0])
                        for p in xrange(p_start, p_end):
                            for q in xrange(q_start, q_end):
                                for n in xrange(FLAGS.num_box):

                                    pm = det[i, j, n]
                                    pc = det[p, q, FLAGS.num_box + n]
                                    pconnm = det[i, j, FLAGS.num_box*6 \
                                                + n*(FLAGS.output_row+FLAGS.output_col) + p] \
                                           * det[i, j, FLAGS.num_box*6 \
                                                + n*(FLAGS.output_row+FLAGS.output_col)+FLAGS.output_row + q]
                                    pconnc = det[p, q, FLAGS.num_box*6 \
                                                + (FLAGS.num_box+n)*(FLAGS.output_row+FLAGS.output_col) + i] \
                                           * det[p, q, FLAGS.num_box*6 \
                                                + (FLAGS.num_box+n)*(FLAGS.output_row+FLAGS.output_col)+FLAGS.output_row + j]
                                    score_part = pm * pc * (pconnm + pconnc) / 2.0
                                    if score_part < 0.001:
                                      continue

                                    xm = (det[i, j, (FLAGS.num_box+n)*2 + 0] + j) / FLAGS.output_col * datum[1][1][0]
                                    ym = (det[i, j, (FLAGS.num_box+n)*2 + 1] + i) / FLAGS.output_row * datum[1][1][1]
                                    xc = (det[p, q, (FLAGS.num_box*2+n)*2 + 0] + q) / FLAGS.output_col * datum[1][1][0]
                                    yc = (det[p, q, (FLAGS.num_box*2+n)*2 + 1] + p) / FLAGS.output_row * datum[1][1][1]
                                    
                                    w_2 = ss[k][0] * (xc - xm)
                                    h_2 = ss[k][1] * (yc - ym)
                                    x1 = xm - w_2; y1 = ym - h_2
                                    x2 = xm + w_2; y2 = ym + h_2
                    
                                    for cls in xrange(FLAGS.num_class):
                                        pclsm = det[i, j, FLAGS.num_box*(6+2*(FLAGS.output_row+FLAGS.output_col)) \
                                                + n*FLAGS.num_class + cls]
                                        pclsc = det[p, q, FLAGS.num_box*(6+2*(FLAGS.output_row+FLAGS.output_col)+FLAGS.num_class) \
                                                + n*FLAGS.num_class + cls]

                                        score = score_part * pclsm * pclsc
                                        if score < 0.001:
                                            continue
                                        boxes[k][cls].append([x1, y1, x2, y2, score])
                
            for k in xrange(4):
                tmp = [np.float32(x) for x in boxes[k]]
                res = apply_nms(tmp, 0.5, int(FLAGS.gpu_list))
                res_list.append(res)
            
            boxes = [boxes[0][i] + boxes[1][i] + boxes[2][i] + boxes[3][i] for i in xrange(FLAGS.num_class)]
            boxes = [np.float32(x) for x in boxes]
            res = apply_nms(boxes, 0.5, int(FLAGS.gpu_list))
            res_list.append(res)
            
            res_list.append(datum[2])
            save_result(datum[0], datum[1], res_list, FLAGS.eval_dir)
        duration = time.time() - start_time
        print '%s: Use time: %d seconds.' %(datetime.now(), duration)
        

def main(unused_argv=None):
    FLAGS(sys.argv)
    time_now = str(datetime.now())
    time_now = time_now.replace(' ', '-').replace(':', '-').replace('.', '-')
    FLAGS.eval_dir = os.path.join(FLAGS.eval_dir, time_now)
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()

if __name__ == '__main__':
    tf.app.run()
