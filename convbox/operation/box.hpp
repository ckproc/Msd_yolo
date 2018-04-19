// --------------------------------------------------------
// ConvBox
// Copyright (c) 2016 HUST
// Licensed under The MIT License [see LICENSE for details]
// Written by Ckb
// --------------------------------------------------------

#ifndef CAFFE_UTIL_BOX_HPP_
#define CAFFE_UTIL_BOX_HPP_

#include <vector>

template <typename Dtype>
Dtype box_iou(std::vector<Dtype> &a, std::vector<Dtype> &b);

template <typename Dtype>
Dtype box_union(std::vector<Dtype> &a, std::vector<Dtype> &b);

template <typename Dtype>
Dtype box_intersection(std::vector<Dtype> &a, std::vector<Dtype> &b);

template <typename Dtype>
Dtype overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2);

template <typename Dtype>
Dtype box_rmse(std::vector<Dtype> &a, std::vector<Dtype> &b);

#endif  // CAFFE_UTIL_BOX_HPP_