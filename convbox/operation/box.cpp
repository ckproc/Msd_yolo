// --------------------------------------------------------
// ConvBox
// Copyright (c) 2016 HUST
// Licensed under The MIT License [see LICENSE for details]
// Written by Ckb
// --------------------------------------------------------

#include <cmath>
#include "box.hpp"
    
using namespace std;

template <typename Dtype>
Dtype box_iou(vector<Dtype> &a, vector<Dtype> &b){
  return box_intersection(a, b) / box_union(a, b);
}

template float box_iou<float>(vector<float> &a, vector<float> &b);
template double box_iou<double>(vector<double> &a, vector<double> &b);

template <typename Dtype>
Dtype box_union(vector<Dtype> &a, vector<Dtype> &b){
  Dtype i = box_intersection(a, b);
  Dtype u = a[2] * a[3] + b[2] * b[3] - i;
  return u;
}

template float box_union<float>(vector<float> &a, vector<float> &b);
template double box_union<double>(vector<double> &a, vector<double> &b);

template <typename Dtype>
Dtype box_intersection(vector<Dtype> &a, vector<Dtype> &b){
  Dtype w = overlap(a[0], a[2], b[0], b[2]);
  Dtype h = overlap(a[1], a[3], b[1], b[3]);
  if (w < 0 || h < 0) return 0;
  Dtype area = w * h;
  return area;
}

template float box_intersection<float>(vector<float> &a, vector<float> &b);
template double box_intersection<double>(vector<double> &a, vector<double> &b);

template <typename Dtype>
Dtype overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2){
  Dtype l1 = x1 - w1 / 2;
  Dtype l2 = x2 - w2 / 2;
  Dtype left = l1 > l2 ? l1 : l2;
  Dtype r1 = x1 + w1 / 2;
  Dtype r2 = x2 + w2 / 2;
  Dtype right = r1 < r2 ? r1 : r2;
  return right - left;
}

template float overlap<float>(float x1, float w1, float x2, float w2);
template double overlap<double>(double x1, double w1, double x2, double w2);

template <typename Dtype>
Dtype box_rmse(vector<Dtype> &a, vector<Dtype> &b){
  return sqrt(pow(a[0] - b[0], 2) + 
              pow(a[1] - b[1], 2) + 
              pow(a[2] - b[2], 2) + 
              pow(a[3] - b[3], 2));
}

template float box_rmse<float>(vector<float> &a, vector<float> &b);
template double box_rmse<double>(vector<double> &a, vector<double> &b);


