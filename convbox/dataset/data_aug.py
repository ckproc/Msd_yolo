# --------------------------------------------------------
# ConvBox
# Copyright (c) 2016 HUST
# Licensed under The MIT License [see LICENSE for details]
# Written by Ckb
# --------------------------------------------------------
from __future__ import division

import random
import numpy as np
from proto.convbox_pb2 import Datum

# global datum
datum = Datum()


def data_aug(raw_datum, data_aug_params):
    if (random.uniform(0, 1) > 0.5):
        cropped, bboxes = load_yolo_data(raw_datum, data_aug_params['yolo'])
    else:
        cropped, bboxes = load_ssd_data(raw_datum, data_aug_params['ssd'])
    return cropped, bboxes


def load_yolo_data(raw_datum, data_aug_params):
    global datum
    datum.ParseFromString(raw_datum)
    orig = np.fromstring(datum.data, dtype=np.uint8)
    orig = orig.reshape(datum.height, datum.width, datum.channels)
    
    #jitter = data_aug_params['jitter']
    
    jitter = 0.2
    ow = datum.width
    oh = datum.height

    dw = int(ow*jitter)
    dh = int(oh*jitter)
    
    #dw_out = int(ow * jitter[0])
    #dw_in  = int(ow * jitter[1])
    #dh_out = int(oh * jitter[0])
    #dh_in  = int(oh * jitter[1])

    pleft   = random.randint(-dw, dw)
    pright  = random.randint(-dw, dw)
    ptop    = random.randint(-dh, dh)
    pbottom = random.randint(-dh, dh)

    swidth = ow - pleft - pright
    sheight = oh - ptop - pbottom
    
    cropped = np.ones((sheight, swidth, datum.channels), dtype=np.uint8)
    cropped *= random.randint(125, 130)
    cropped[max(-ptop, 0):min(sheight + pbottom, sheight), max(-pleft, 0):min(swidth + pright, swidth)] \
            = orig[max(ptop, 0):min(oh - pbottom, oh), max(pleft, 0):min(ow - pright, ow)]

    flip = (random.uniform(0, 1) > 0.5)
    if flip:
        cropped = cropped[:, ::-1, :]

    sx = float(swidth) / ow
    sy = float(sheight) / oh

    dx = (float(pleft) / ow) / sx
    dy = (float(ptop) / oh) / sy
    
    # correct bboxes: [x, y, w, h]
    bboxes = []
    for box in datum.box:
        left   = (box.x - box.w / 2.0) / sx - dx
        #left   = (box.x - box.w / 2.0)
        right  = (box.x + box.w / 2.0) / sx - dx
        #right  = (box.x + box.w / 2.0)
        top    = (box.y - box.h / 2.0) / sy - dy
        #top    = (box.y - box.h / 2.0)
        bottom = (box.y + box.h / 2.0) / sy - dy
        #bottom = (box.y + box.h / 2.0)

        if flip:
            #orig = orig[:,::-1,:]
            swap = left
            left = 1.0 - right
            right = 1.0 - swap
        
        left   = constrain(0.0, 1.0, left)
        right  = constrain(0.0, 1.0, right)
        top    = constrain(0.0, 1.0, top)
        bottom = constrain(0.0, 1.0, bottom)
        
        x = (left + right) / 2.0
        y = (top + bottom) / 2.0
        w = right - left
        h = bottom - top
        
        bboxes.append([box.id, x, y, w, h])

    return cropped, bboxes


def load_ssd_data(raw_datum, data_aug_params):
    global datum
    datum.ParseFromString(raw_datum)
    orig = np.fromstring(datum.data, dtype=np.uint8)
    orig = orig.reshape(datum.height, datum.width, datum.channels)

    sbox_param = data_aug_params['sample_box']

    # read boxes: [x1, y1, x2, y2]
    bboxes = []
    for box in datum.box:
        x1 = box.x - box.w / 2.0
        x2 = box.x + box.w / 2.0
        y1 = box.y - box.h / 2.0
        y2 = box.y + box.h / 2.0
        bbox = [box.id, x1, y1, x2, y2]
        bboxes.append(bbox)

    sampleBoxes = []
    sampleBoxes.append([0.0, 0.0, 1.0, 1.0])

    max_jaccard_overlap = [1.0]
    min_jaccard_overlap = [0.1, 0.3, 0.5, 0.7, 0.9]
    for min_overlap in min_jaccard_overlap:
        found = False
        for i in range(50):
            if found:
                break
            
            samplebox = SampleBBox(sbox_param[0], sbox_param[1], sbox_param[2], sbox_param[3])
            if SatisfySampleConstraint(samplebox, bboxes, min_overlap):
                found = True
                sampleBoxes.append(samplebox)
                
    for max_overlap in max_jaccard_overlap:
        found = False
        for i in range(50):
            if found:
                break
            
            samplebox = SampleBBox(sbox_param[0], sbox_param[1], sbox_param[2], sbox_param[3])
            if SatisfySampleConstraint2(samplebox, bboxes, max_overlap):
                found = True
                sampleBoxes.append(samplebox)

    cropped = 0
    cropped_bboxes = []
    if len(sampleBoxes) > 0:
        rand_idx = random.randint(0, len(sampleBoxes) - 1)
        crop_bbox = sampleBoxes[rand_idx]
        crop_bbox = ClipBBox(crop_bbox)
        left  = int(crop_bbox[0] * datum.width)
        upper = int(crop_bbox[1] * datum.height)
        right = int(crop_bbox[2] * datum.width)
        lower = int(crop_bbox[3] * datum.height)
        cropped = orig[upper:lower, left:right]
        
        for bbox in bboxes:
            if (not MeetEmitConstraint(crop_bbox, bbox[1:])):
                continue
            
            flag, proj_bbox = ProjectBBox(crop_bbox, bbox[1:])
            
            if flag:
                x = (proj_bbox[0] + proj_bbox[2]) / 2.0
                y = (proj_bbox[1] + proj_bbox[3]) / 2.0
                w = proj_bbox[2] - proj_bbox[0]
                h = proj_bbox[3] - proj_bbox[1]
                cropped_bboxes.append([bbox[0], x, y, w, h])
    else:
        cropped = orig
        for box in datum.box:
            bbox = [box.id, box.x, box.y, box.w, box.h]
            cropped_bboxes.append(bbox)

    flip = (random.uniform(0, 1) > 0.5)
    if flip:
        cropped = cropped[:, ::-1, :]
        for bbox in cropped_bboxes:
            bbox[1] = 1.0 - bbox[1]
    
    return cropped, cropped_bboxes
        

def SampleBBox(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio):
    scale = random.uniform(min_scale, max_scale)
    minAR = max(min_aspect_ratio, scale ** 2)
    maxAR = min(max_aspect_ratio, 1.0/(scale ** 2))
    aspect_ratio = random.uniform(minAR, maxAR)

    bbox_width = scale * (aspect_ratio ** 0.5)
    bbox_height = scale / (aspect_ratio ** 0.5)

    w_off = random.uniform(0, 1 - bbox_width)
    h_off = random.uniform(0, 1 - bbox_height)

    samplebox = [w_off, h_off, w_off + bbox_width, h_off + bbox_height]

    return samplebox

def SatisfySampleConstraint(samplebox, bboxes, min_overlap):
    Found = False
    for bbox in bboxes:
        jaccard_overlap = JaccardOverlap(samplebox, bbox[1:])
        if jaccard_overlap < min_overlap:
            continue
        Found = True
        break 

    return Found

def SatisfySampleConstraint2(samplebox, bboxes, max_overlap):
    Found = False
    for bbox in bboxes:
        jaccard_overlap = JaccardOverlap(samplebox, bbox[1:])
        if jaccard_overlap > max_overlap:
            continue
        Found = True
        break 

    return Found

def JaccardOverlap(a, b):
    return BBoxIntersection(a, b) / BBoxUnion(a, b)

def BBoxUnion(a, b):
    i =  BBoxIntersection(a, b)
    u = BBoxSize(a) + BBoxSize(b) - i
    return u

def BBoxIntersection(a, b):
    w = Overlap(a[0], a[2], b[0], b[2])
    h = Overlap(a[1], a[3], b[1], b[3])
    if (w < 0 or h < 0):
        return 0
    area = w * h
    return area

def Overlap(l1, r1, l2, r2):
    left = max(l1, l2)
    right = min(r1, r2)
    return right - left

def ClipBBox(bbox):
    bbox[0] = constrain(0.0, 1.0, bbox[0])
    bbox[1] = constrain(0.0, 1.0, bbox[1])
    bbox[2] = constrain(0.0, 1.0, bbox[2])
    bbox[3] = constrain(0.0, 1.0, bbox[3])
    return bbox

def MeetEmitConstraint(src_bbox, bbox):
    x_center = (bbox[0] + bbox[2]) / 2.0
    y_center = (bbox[1] + bbox[3]) / 2.0

    if (x_center >= src_bbox[0] and x_center <= src_bbox[2] and y_center >= src_bbox[1] and y_center <= src_bbox[3]):
        return True
    else:
        return False

def ProjectBBox(src_bbox, bbox):
    if (bbox[0] >= src_bbox[2] or bbox[2] <= src_bbox[0] or bbox[1] >= src_bbox[3] or bbox[3] <= src_bbox[0]):
        return False, [0, 0, 0, 0]

    src_width = src_bbox[2] - src_bbox[0]
    src_height = src_bbox[3] - src_bbox[1]

    xmin = (bbox[0] - src_bbox[0]) / src_width
    ymin = (bbox[1] - src_bbox[1]) / src_height
    xmax = (bbox[2] - src_bbox[0]) / src_width
    ymax = (bbox[3] - src_bbox[1]) / src_height

    proj_box = [xmin, ymin, xmax, ymax]
    proj_box = ClipBBox(proj_box)

    if BBoxSize(proj_box) > 0:
        return True, proj_box
    else:
        return False, [0, 0, 0, 0]

def BBoxSize(bbox):
    if (bbox[2] < bbox[0] or bbox[3] < bbox[1]):
        return 0
    else:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width * height
    
def constrain(minv, maxv, a):
    if a < minv:
        return minv
    if a > maxv:
        return maxv
    return a
	
def randscale(s):
    scale = random.uniform(1.0, s)
    if random.uniform(0, 1) > 0.5:
        return scale
    else:
        return 1.0 / scale