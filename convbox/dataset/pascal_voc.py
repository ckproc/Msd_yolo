# --------------------------------------------------------
# ConvBox
# Copyright (c) 2016 HUST
# Licensed under The MIT License [see LICENSE for details]
# Written by Ckb
# --------------------------------------------------------
from __future__ import division

import cv2
import os.path
import random
import itertools
import numpy as np
from random import shuffle
from datetime import datetime
from dataset import dataset
from data_aug import data_aug
from proto.convbox_pb2 import Datum
import math
CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
 "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
 "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


class pascal_voc(dataset):
    """
    pascal voc dataset.
    """
    def __init__(self, params):
        dataset.__init__(self, params)
        self._input_shape = params['input_shape']
        self._output_shape = params['output_shape']
        self._num_class = params['num_class']
        assert self._num_class == 20
        self.data_aug_params = {'yolo':{'jitter':[-0.3, 0.1]}, 
                                 'ssd':{'sample_box':[0.3, 1.0, 0.5, 2.0]} }
        print '{}: {} initialized with input_shape: {}, output_shape: {}, num_class: {}.'.format(
                datetime.now(), self.name, self.input_shape, self.output_shape, self.num_class)

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def num_class(self):
        return self._num_class

    @property
    def class_list(self):
        return CLASSES

    @property
    def train_datum_iterator(self):
        for _ in itertools.count(): 
            yield self.next_train_datum

    @property
    def next_train_datum(self):
        raw_datum = self.next_raw_datum
        image, bboxes = data_aug(raw_datum, self.data_aug_params)
        sized = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))

        #channels = 3 + self.output_shape[0] + self.output_shape[1] + self.num_class
        channels = 5 + self.num_class
        shape = (self.output_shape[0], self.output_shape[1], channels)
        #shape = (self.output_shape[0], self.output_shape[1], channels * 2)
        #multilabels = [np.zeros(shape).astype(np.float32) for _ in xrange(4)]
        multilabels = np.zeros(shape).astype(np.float32)

        # set label
        # bboxes: [x, y, w, h]
        rand_list = range(len(bboxes))    
        #ss = [(-2, -2), (+2, -2), (-2, +2), (+2, +2)]
        #for k in xrange(4):
        shuffle(rand_list)
        for i in rand_list:
            if (bboxes[i][3] < 0.01 or bboxes[i][4] < 0.01):
                continue
        
            #cat, xm, ym = bboxes[i][:3]
            cat,x,y,w,h = bboxes[i][:5]
            #xc = bboxes[i][1] + bboxes[i][3] / ss[k][0]
            #xc = bboxes[i][1] + bboxes[i][3] / ss[0][0]
            #yc = bboxes[i][2] + bboxes[i][4] / ss[k][1]
            #yc = bboxes[i][2] + bboxes[i][4] / ss[0][1]
            
            col = int(x*14)
            row = int(y*14)
            
            x = x*14-col    #one grid
            y = y*14-row
            w = math.sqrt(w)
            h = math.sqrt(h)
            
            
            #colm = int(xm * self.output_shape[1])
            #rowm = int(ym * self.output_shape[0])
            #colc = int(xc * self.output_shape[1])
            #rowc = int(yc * self.output_shape[0])

            #if colm == self.output_shape[1]: colm -= 1
            #if rowm == self.output_shape[0]: rowm -= 1
            #if colc == self.output_shape[1]: colc -= 1
            #if rowc == self.output_shape[0]: rowc -= 1
        
            if (multilabels[row, col, 0] != 0):
                continue
        
            #xm = xm * self.output_shape[1] - colm
            #ym = ym * self.output_shape[0] - rowm
            #xc = xc * self.output_shape[1] - colc
            #yc = yc * self.output_shape[0] - rowc 
            
            multilabels[row, col, 0] = 1.0
            multilabels[row, col, cat+1] = 1.0
            multilabels[row, col, 21] = x
            multilabels[row, col, 22] = y
            multilabels[row, col, 23] = w
            multilabels[row, col, 24] = h
            
            
  
        return transformer(sized), multilabels
        #return sized, multilabels


    def eval_datum_iterator(self, begin=0, length=0):
        if length == 0:
            length = self.num_images

        datum = Datum()
        txn = self._env.begin()
        for k in self._lmdb_keys[begin: begin + length]:
            raw_datum = txn.get(k)
            datum.ParseFromString(raw_datum)
            image = np.fromstring(datum.data, dtype=np.uint8)
            image = image.reshape(datum.height, datum.width, datum.channels)
            sized = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
            sized = transformer(sized)
            sized = sized.reshape((1, self.input_shape[0], self.input_shape[1], 3))

            bboxes = [[] for _ in xrange(self.num_class)]
            for box in datum.box:
                x = box.x * datum.width
                y = box.y * datum.height
                w = box.w * datum.width
                h = box.h * datum.height
                bbox = [x-w/2, y-h/2, x+w/2, y+h/2, 1.0]
                bboxes[box.id].append(bbox)
                
            bboxes = [np.float32(x) for x in bboxes]
            yield (datum.id, (sized, (datum.width, datum.height)), bboxes)

    def save_result(self, datum_id, im_size, boxes, eval_dir):
        for key, value in enumerate(boxes):
            filename = 'comp4_det_test_' + CLASSES[key] + '.txt'
            fp = open(os.path.join(eval_dir, filename), 'a')

            for box in value:
                [x1, y1, x2, y2] = box[:-1]
                
                if x1 < 0: x1 = 0
                if y1 < 0: y1 = 0
                if x2 > im_size[0] - 1: x2 = im_size[0] - 1
                if y2 > im_size[1] - 1: y2 = im_size[1] - 1

                fp.write(str(datum_id) + ' ' + str(box[4]) + ' ')
                fp.write(str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2))
                fp.write('\n')

            fp.close()

def transformer(im):
    im = np.float32(im)
    im *= (2.0 / 255)
    im -= 1.0
    return im
