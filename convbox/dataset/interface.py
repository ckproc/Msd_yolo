# --------------------------------------------------------
# ConvBox
# Copyright (c) 2016 HUST
# Licensed under The MIT License [see LICENSE for details]
# Written by Ckb
# --------------------------------------------------------

from pascal_voc import pascal_voc
from coco import coco
from kitti import kitti

def dataset(params):
    if params['name'] == 'pascal_voc':
        return pascal_voc(params)
    elif params['name'] == 'coco':
        return coco(params)
    elif params['name'] == 'kitti':
        return kitti(params)
    else:
        raise ValueError('Dataset [%s] was not recognized.', params['name'])
