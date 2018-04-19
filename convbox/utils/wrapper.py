# --------------------------------------------------------
# ConvBox
# Copyright (c) 2016 HUST
# Licensed under The MIT License [see LICENSE for details]
# Written by Ckb
# --------------------------------------------------------

from nms.gpu_nms import gpu_nms
from nms.cpu_nms import cpu_nms
from tensor_to_box.py_cpu_tensor_to_box import py_cpu_tensor_to_box


def nms(dets, thresh, device_id=0, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if not force_cpu:
        return gpu_nms(dets, thresh, device_id)
    else:
        return cpu_nms(dets, thresh)

def tensor_to_box(det_list, num_box, im_size, pool=None, device_id=0, force_cpu=False):
    return py_cpu_tensor_to_box(det_list, num_box, im_size, pool)
