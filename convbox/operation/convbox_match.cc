// --------------------------------------------------------
// ConvBox
// Copyright (c) 2016 HUST
// Licensed under The MIT License [see LICENSE for details]
// Written by Ckb
// --------------------------------------------------------

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include "box.hpp"

using namespace tensorflow;
using std::vector;
using std::pow;
using std::max;
using std::abs;

REGISTER_OP("ConvboxMatch")
    .Attr("T: {float, double}")
    .Input("pred: T")
    .Input("label: T")
    .Input("num_box: int32")
    .Input("num_class: int32")
    .Output("pred_mask: bool")
    .Output("label_mask: bool")
    .Output("update_label: T");
    

template <typename T>    
class ConvboxMatchOp : public OpKernel {
 public:
  explicit ConvboxMatchOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
      // Grab the input tensor
      const Tensor& pred_T = context->input(0);
      const Tensor& label_T = context->input(1);
      const Tensor& num_box_T = context->input(2);
      const Tensor& num_class_T = context->input(3);
      auto pred = pred_T.tensor<T, 4>();
      auto label = label_T.tensor<T, 4>();
      auto num_box = num_box_T.scalar<int>()(0);
      auto num_class = num_class_T.scalar<int>()(0);
      
      Tensor* pred_mask_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, pred_T.shape(), &pred_mask_tensor));
      auto pred_mask_output = pred_mask_tensor->tensor<bool, 4>();
      
      Tensor* label_mask_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(1, label_T.shape(), &label_mask_tensor));
      auto label_mask_output = label_mask_tensor->tensor<bool, 4>();

      Tensor* update_label_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(2, label_T.shape(), &update_label_tensor));
      auto update_label_output = update_label_tensor->tensor<T, 4>();
      
      assert(num_box == 2);
      
      T avg_iou =0;
      T avg_dist = 0;
      T avg_cat = 0;
      T avg_allcat = 0;
      T avg_conn = 0;
      T avg_allconn = 0;
      T avg_obj = 0;
      T avg_anyobj = 0;
      int count = 0;
      
      int batch_size = label_T.dim_size(0);
      int rows = label_T.dim_size(1);
      int cols = label_T.dim_size(2);
      int channels = label_T.dim_size(3);
      // assert(rows == 14);
      // assert(cols == 14);
      for (int batch = 0; batch < batch_size; ++batch){
        for (int rowm = 0; rowm < rows; ++rowm){
          for (int colm = 0; colm < cols; ++colm){
            for (int c = 0; c < channels; ++c){
              for (int i = 0; i < num_box; ++i)
                pred_mask_output(batch, rowm, colm, c*num_box+i) = false;
                //pred_mask_output(batch, rowm, colm, channels*num_box+c) = false;
              label_mask_output(batch, rowm, colm, c) = false;
              update_label_output(batch, rowm, colm, c) = label(batch, rowm, colm, c);
            }
          }
        }
          
        for (int rowm = 0; rowm < rows; ++rowm){
          for (int colm = 0; colm < cols; ++colm){
                        
            for (int i = 0; i < num_box; ++i)
              avg_anyobj += pred(batch, rowm, colm, i);
            
            if (label(batch, rowm, colm, 0)){
              // std::cout << best_index << std::endl;
              T xm = (label(batch, rowm, colm, num_class+1) + colm) / cols;
              //T xm = (label(batch, rowm, colm, 2) + colm) / cols;
              T ym = (label(batch, rowm, colm, num_class+2) + rowm) / rows;
              //T ym = (label(batch, rowm, colm, 3) + rowm) / rows;
              T wm = pow(label(batch,rowm,colm,num_class+3),2);
              T hm = pow(label(batch,rowm,colm,num_class+4),2);
              //T xc = (label(batch, rowc, colc, 4) + colc) / cols;
              //T yc = (label(batch, rowc, colc, 5) + rowc) / rows;
              vector<T> truth(4);
              truth[0] = xm; truth[1] = ym;
              //truth[2] = fabs(xm - xc) * 2.0;
              truth[2] = wm;
              //truth[3] = fabs(ym - yc) * 2.0;
              truth[3] = hm;
              int best_index = -1;
              T best_iou = 0;
              T best_rmse = 10000;
              //T best_match = -100;
              for(int i=0;i<num_box;i++){
                T xp = ( pred(batch, rowm, colm, num_box*(num_class+1)+4*i) + colm) / cols;
                T yp = (pred(batch, rowm, colm, num_box*(num_class+1)+4*i+1) + rowm) / rows;
                T wp = pow(pred(batch,rowm,colm,num_box*(num_class+1)+4*i+2),2);
                T hp = pow(pred(batch,rowm,colm,num_box*(num_class+1)+4*i+3),2);
                vector<T> out(4);
                out[0] = xp; out[1] = yp;
                out[2] = wp;
                out[3] = hp;
                T iou = box_iou(out, truth);
                T rmse = box_rmse(out, truth);
                //std::cout<<"current iou"<<iou<<"  rmse:"<<rmse<<std::endl; 
                if(best_iou > 0 || iou>0){
                    if(iou>best_iou){
                        best_iou = iou;
                        best_index = i;
                    }        
                }
                else{
                    if(rmse<best_rmse){
                        best_rmse = rmse;
                        best_index = i;
                    }
                }                    
              }
              vector<T> best_out(4);
              T xp = ( pred(batch, rowm, colm, num_box*(num_class+1)+4*best_index) + colm) / cols;
              T yp = (pred(batch, rowm, colm, num_box*(num_class+1)+4*best_index+1) + rowm) / rows;
              T wp = pow(pred(batch,rowm,colm,num_box*(num_class+1)+4*best_index+2),2);
              T hp = pow(pred(batch,rowm,colm,num_box*(num_class+1)+4*best_index+3),2);
              best_out[0] = xp;
              best_out[1] = yp;
              best_out[2] = wp;
              best_out[3] = hp;
              T curr_iou = box_iou(best_out,truth);
              avg_iou +=curr_iou;
              /*
              for (int i = 0; i < num_box; ++i){
                
                T pred_cat = 0.0;
                for (int j = 0; j < num_class; ++j){
                  T  pred_cat_m =  pred(batch, rowm, colm, (6+2*(rows+cols))*num_box+i*num_class+j);
                  T  pred_cat_c =  pred(batch, rowc, colc, (6+2*(rows+cols)+num_class)*num_box+i*num_class+j);
                  T label_cat_m = label(batch, rowm, colm, 6+2*(rows+cols)+j);
                  T label_cat_c = label(batch, rowc, colc, 6+2*(rows+cols)+num_class+j);
                  pred_cat += pow(pred_cat_m - label_cat_m, 2);
                  pred_cat += pow(pred_cat_c - label_cat_c, 2); 
                }

                T pred_conn_m = pred(batch, rowm, colm, 6*num_box+i*(rows+cols)+rowc)
                              * pred(batch, rowm, colm, 6*num_box+i*(rows+cols)+rows+colc);
                T pred_conn_c = pred(batch, rowc, colc, (6+rows+cols)*num_box+i*(rows+cols)+rowm)
                              * pred(batch, rowc, colc, (6+rows+cols)*num_box+i*(rows+cols)+rows+colm);
                T pred_conn = (pred_conn_m + pred_conn_c) / 2.0;
                
                xm = (pred(batch, rowm, colm, (num_box+i)*2+0) + colm) / cols;
                ym = (pred(batch, rowm, colm, (num_box+i)*2+1) + rowm) / rows;
                xc = (pred(batch, rowc, colc, (num_box*2+i)*2+0) + colc) / cols;
                yc = (pred(batch, rowc, colc, (num_box*2+i)*2+1) + rowc) / rows;
                
                
                vector<T> out(4);
                out[0] = xm; out[1] = ym;
                out[2] = fabs(xm - xc) * 2.0;
                out[3] = fabs(ym - yc) * 2.0;
                T iou = box_iou(out, truth);
                T rmse = box_rmse(out, truth);
                
                T cur_match = pred_conn * (iou - rmse + 0.1) + 0.1 * (2 - pred_cat);
                
                if (cur_match > best_match){
                  best_match = cur_match;
                  best_index = i;
                }
              }*/
              assert(best_index != -1);
              //assert(rowc != -1);
              //assert(colc != -1);
              //vector<int> row = {rowm, rowc};
              //vector<int> col = {colm, colc};
              pred_mask_output(batch, rowm, colm, best_index) = true;
              label_mask_output(batch,rowm,colm,0) =true;
              avg_obj += pred(batch, rowm, colm, best_index);
              //std::cout<<"bestindex:"<<best_index<<"pos:"<<pred(batch, rowm, colm, best_index)<<std::endl;
              for(int i=0;i<num_class;i++){
                pred_mask_output(batch,rowm,colm,num_box+num_class*best_index+i)=true;
                label_mask_output(batch,rowm,colm,1+i)=true;
                if (label(batch, rowm, colm, 1+i) == 1)
                    avg_cat += pred(batch, rowm, colm, num_box+num_class*best_index+i);
                avg_allcat += pred(batch, rowm, colm, num_box+num_class*best_index+i);
              }
              for(int i=0;i<4;i++){
                 pred_mask_output(batch,rowm,colm,num_box*(num_class+1)+4*best_index+i)=true;
                 label_mask_output(batch,rowm,colm,num_class+1+i)=true;
              }
              count+=1;
              /*
              for (int n = 0; n < 2; ++n) {
                pred_mask_output(batch, row[n], col[n], n*num_box+best_index) = true;
                label_mask_output(batch, row[n], col[n], n) = true;
                avg_obj += pred(batch, row[n], col[n], n*num_box+best_index);
                
                pred_mask_output(batch, row[n], col[n], ((1+n)*num_box+best_index)*2+0) = true;
                pred_mask_output(batch, row[n], col[n], ((1+n)*num_box+best_index)*2+1) = true;
                label_mask_output(batch, row[n], col[n], 2*(1+n)+0) = true;
                label_mask_output(batch, row[n], col[n], 2*(1+n)+1) = true;
                avg_dist += (pow(pred(batch, row[n], col[n], ((1+n)*num_box+best_index)*2+0) - label(batch, row[n], col[n], 2*(1+n)+0), 2)
                           + pow(pred(batch, row[n], col[n], ((1+n)*num_box+best_index)*2+1) - label(batch, row[n], col[n], 2*(1+n)+1), 2));
                
                for (int i = 0; i < (rows+cols); ++i){
                  pred_mask_output(batch, row[n], col[n], (6+n*(rows+cols))*num_box+best_index*(rows+cols)+i) = true;
                  label_mask_output(batch, row[n], col[n], 6+n*(rows+cols)+i) = true;
                  if (label(batch, row[n], col[n], 6+n*(rows+cols)+i) == 1)
                    avg_conn += pred(batch, row[n], col[n], (6+n*(rows+cols))*num_box+best_index*(rows+cols)+i);
                  avg_allconn += pred(batch, row[n], col[n], (6+n*(rows+cols))*num_box+best_index*(rows+cols)+i);
                }
                
                for (int i = 0; i < num_class; ++i){
                  pred_mask_output(batch, row[n], col[n], (6+2*(rows+cols)+n*num_class)*num_box+best_index*num_class+i) = true;
                  label_mask_output(batch, row[n], col[n], 6+2*(rows+cols)+n*num_class+i) = true;
                  if (label(batch, row[n], col[n], 6+2*(rows+cols)+n*num_class+i) == 1)
                    avg_cat += pred(batch, row[n], col[n], (6+2*(rows+cols)+n*num_class)*num_box+best_index*num_class+i);
                  avg_allcat += pred(batch, row[n], col[n], (6+2*(rows+cols)+n*num_class)*num_box+best_index*num_class+i);
                }
                
                count += 1;
              }
              */
              
            } // has obj
          } // colm
        } // rowm
      } // batch
      
      if (count > 0){
        std::cout << std::fixed  
                <<" Detection Iou "<<avg_iou/count
                << ", Pos Cat: " << avg_cat/count
                << ", All Cat: " << avg_allcat/(count*num_class) 
                << ", Pos Obj: " << avg_obj/count
                << ", Any Obj: " << avg_anyobj/(batch_size*rows*cols*num_box*2)
                << ", total count: " << count << std::endl;
      }
      

  }
};

REGISTER_KERNEL_BUILDER(Name("ConvboxMatch").Device(DEVICE_CPU).TypeConstraint<float>("T"), ConvboxMatchOp<float>);
REGISTER_KERNEL_BUILDER(Name("ConvboxMatch").Device(DEVICE_CPU).TypeConstraint<double>("T"), ConvboxMatchOp<double>);
