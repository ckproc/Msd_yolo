import os
import tensorflow as tf
import numpy as np
import cPickle as pickle
import torchfile  # pip install torchfile
import sys



from convbox_model import inference
T7_PATH = 'msdnet--step=4--block=5--growthRate=16.t7'
o = torchfile.load(T7_PATH)
INIT_CHECKPOINT_DIR = './init'
print('Load weights in a brute-force way')
Joint=o.modules[0].modules    #nn.JointTrainMudule

block1 = Joint[0].modules   #first block   1+4
MSDNet_Layer_first = block1[0].modules

Msdnet_b0_scale0_conv1_weights = MSDNet_Layer_first[0].modules[0].weight   #(32, 3, 7, 7)
Msdnet_b0_scale0_conv1_BatchNorm_beta =MSDNet_Layer_first[0].modules[1].bias   #(32,)
Msdnet_b0_scale0_conv1_BatchNorm_gamma =MSDNet_Layer_first[0].modules[1].weight #(32,)

Msdnet_b0_scale1_conv1_weights =MSDNet_Layer_first[1].modules[0].weight    #(64, 32, 3, 3)
Msdnet_b0_scale1_conv1_BatchNorm_beta =MSDNet_Layer_first[1].modules[1].bias
Msdnet_b0_scale1_conv1_BatchNorm_gamma =MSDNet_Layer_first[1].modules[1].weight

Msdnet_b0_scale2_conv1_weights =MSDNet_Layer_first[2].modules[0].weight
Msdnet_b0_scale2_conv1_BatchNorm_beta =MSDNet_Layer_first[2].modules[1].bias
Msdnet_b0_scale2_conv1_BatchNorm_gamma =MSDNet_Layer_first[2].modules[1].weight

Msdnet_b0_scale3_conv1_weights =MSDNet_Layer_first[3].modules[0].weight
Msdnet_b0_scale3_conv1_BatchNorm_beta =MSDNet_Layer_first[3].modules[1].bias
Msdnet_b0_scale3_conv1_BatchNorm_gamma =MSDNet_Layer_first[3].modules[1].weight

MSDNet_Layer1_1 = block1[1].modules
Msdnet_b0_step0_scale0_normal_convbottleneck_weights = MSDNet_Layer1_1[0].modules[0].modules[1].modules[0].weight #(16, 32, 1, 1)
Msdnet_b0_step0_scale0_normal_convbottleneck_BatchNorm_beta =MSDNet_Layer1_1[0].modules[0].modules[1].modules[1].bias  #(16,)
Msdnet_b0_step0_scale0_normal_convbottleneck_BatchNorm_gamma =MSDNet_Layer1_1[0].modules[0].modules[1].modules[1].weight
Msdnet_b0_step0_scale0_normal_convnormal_weights =MSDNet_Layer1_1[0].modules[0].modules[1].modules[3].weight  #(16, 16, 3, 3)
Msdnet_b0_step0_scale0_normal_convnormal_BatchNorm_beta =MSDNet_Layer1_1[0].modules[0].modules[1].modules[4].bias #(16,)
Msdnet_b0_step0_scale0_normal_convnormal_BatchNorm_gamma =MSDNet_Layer1_1[0].modules[0].modules[1].modules[4].weight

Msdnet_b0_step0_scale1_down_convbottleneck_weights = MSDNet_Layer1_1[1].modules[0].modules[1].modules[0].weight
Msdnet_b0_step0_scale1_down_convbottleneck_BatchNorm_beta = MSDNet_Layer1_1[1].modules[0].modules[1].modules[1].bias
Msdnet_b0_step0_scale1_down_convbottleneck_BatchNorm_gamma =MSDNet_Layer1_1[1].modules[0].modules[1].modules[1].weight
Msdnet_b0_step0_scale1_down_convdowm_weights = MSDNet_Layer1_1[1].modules[0].modules[1].modules[3].weight   #(16, 16, 3, 3)
Msdnet_b0_step0_scale1_down_convdowm_BatchNorm_beta =MSDNet_Layer1_1[1].modules[0].modules[1].modules[4].bias
Msdnet_b0_step0_scale1_down_convdowm_BatchNorm_gamma =MSDNet_Layer1_1[1].modules[0].modules[1].modules[4].weight
Msdnet_b0_step0_scale1_normal_convbottleneck_weights = MSDNet_Layer1_1[1].modules[0].modules[2].modules[0].weight  #(32, 64, 1, 1)
Msdnet_b0_step0_scale1_normal_convbottleneck_BatchNorm_beta =MSDNet_Layer1_1[1].modules[0].modules[2].modules[1].bias
Msdnet_b0_step0_scale1_normal_convbottleneck_BatchNorm_gamma =MSDNet_Layer1_1[1].modules[0].modules[2].modules[1].weight
Msdnet_b0_step0_scale1_normal_convnormal_weights =MSDNet_Layer1_1[1].modules[0].modules[2].modules[3].weight   #(16, 32, 3, 3)
Msdnet_b0_step0_scale1_normal_convnormal_BatchNorm_beta =MSDNet_Layer1_1[1].modules[0].modules[2].modules[4].bias
Msdnet_b0_step0_scale1_normal_convnormal_BatchNorm_gamma =MSDNet_Layer1_1[1].modules[0].modules[2].modules[4].weight

Msdnet_b0_step0_scale2_down_convbottleneck_weights = MSDNet_Layer1_1[2].modules[0].modules[1].modules[0].weight
Msdnet_b0_step0_scale2_down_convbottleneck_BatchNorm_beta =MSDNet_Layer1_1[2].modules[0].modules[1].modules[1].bias
Msdnet_b0_step0_scale2_down_convbottleneck_BatchNorm_gamma =MSDNet_Layer1_1[2].modules[0].modules[1].modules[1].weight
Msdnet_b0_step0_scale2_down_convdowm_weights = MSDNet_Layer1_1[2].modules[0].modules[1].modules[3].weight
Msdnet_b0_step0_scale2_down_convdowm_BatchNorm_beta =MSDNet_Layer1_1[2].modules[0].modules[1].modules[4].bias
Msdnet_b0_step0_scale2_down_convdowm_BatchNorm_gamma =MSDNet_Layer1_1[2].modules[0].modules[1].modules[4].weight

Msdnet_b0_step0_scale2_normal_convbottleneck_weights =MSDNet_Layer1_1[2].modules[0].modules[2].modules[0].weight
Msdnet_b0_step0_scale2_normal_convbottleneck_BatchNorm_beta =MSDNet_Layer1_1[2].modules[0].modules[2].modules[1].bias
Msdnet_b0_step0_scale2_normal_convbottleneck_BatchNorm_gamma =MSDNet_Layer1_1[2].modules[0].modules[2].modules[1].weight
Msdnet_b0_step0_scale2_normal_convnormal_weights =MSDNet_Layer1_1[2].modules[0].modules[2].modules[3].weight   #(32, 128, 3, 3)
Msdnet_b0_step0_scale2_normal_convnormal_BatchNorm_beta =MSDNet_Layer1_1[2].modules[0].modules[2].modules[4].bias
Msdnet_b0_step0_scale2_normal_convnormal_BatchNorm_gamma =MSDNet_Layer1_1[2].modules[0].modules[2].modules[4].weight

Msdnet_b0_step0_scale3_down_convbottleneck_weights = MSDNet_Layer1_1[3].modules[0].modules[1].modules[0].weight   #(128, 128, 1, 1)
Msdnet_b0_step0_scale3_down_convbottleneck_BatchNorm_beta =MSDNet_Layer1_1[3].modules[0].modules[1].modules[1].bias
Msdnet_b0_step0_scale3_down_convbottleneck_BatchNorm_gamma =MSDNet_Layer1_1[3].modules[0].modules[1].modules[1].weight
Msdnet_b0_step0_scale3_down_convdowm_weights =MSDNet_Layer1_1[3].modules[0].modules[1].modules[3].weight
Msdnet_b0_step0_scale3_down_convdowm_BatchNorm_beta =MSDNet_Layer1_1[3].modules[0].modules[1].modules[4].bias
Msdnet_b0_step0_scale3_down_convdowm_BatchNorm_gamma =MSDNet_Layer1_1[3].modules[0].modules[1].modules[4].weight

Msdnet_b0_step0_scale3_normal_convbottleneck_weights =MSDNet_Layer1_1[3].modules[0].modules[2].modules[0].weight
Msdnet_b0_step0_scale3_normal_convbottleneck_BatchNorm_beta =MSDNet_Layer1_1[3].modules[0].modules[2].modules[1].bias
Msdnet_b0_step0_scale3_normal_convbottleneck_BatchNorm_gamma =MSDNet_Layer1_1[3].modules[0].modules[2].modules[1].weight
Msdnet_b0_step0_scale3_normal_convnormal_weights =MSDNet_Layer1_1[3].modules[0].modules[2].modules[3].weight  #(32, 128, 3, 3)
Msdnet_b0_step0_scale3_normal_convnormal_BatchNorm_beta =MSDNet_Layer1_1[3].modules[0].modules[2].modules[4].bias
Msdnet_b0_step0_scale3_normal_convnormal_BatchNorm_gamma =MSDNet_Layer1_1[3].modules[0].modules[2].modules[4].weight

MSDNet_Layer1_2 = block1[2].modules
Msdnet_b0_step1_scale0_normal_convbottleneck_weights =        MSDNet_Layer1_2[0].modules[0].modules[1].modules[0].weight #(16, 48, 1, 1)
Msdnet_b0_step1_scale0_normal_convbottleneck_BatchNorm_beta = MSDNet_Layer1_2[0].modules[0].modules[1].modules[1].bias
Msdnet_b0_step1_scale0_normal_convbottleneck_BatchNorm_gamma =MSDNet_Layer1_2[0].modules[0].modules[1].modules[1].weight
Msdnet_b0_step1_scale0_normal_convnormal_weights =            MSDNet_Layer1_2[0].modules[0].modules[1].modules[3].weight
Msdnet_b0_step1_scale0_normal_convnormal_BatchNorm_beta =     MSDNet_Layer1_2[0].modules[0].modules[1].modules[4].bias
Msdnet_b0_step1_scale0_normal_convnormal_BatchNorm_gamma =    MSDNet_Layer1_2[0].modules[0].modules[1].modules[4].weight

Msdnet_b0_step1_scale1_down_convbottleneck_weights =          MSDNet_Layer1_2[1].modules[0].modules[1].modules[0].weight
Msdnet_b0_step1_scale1_down_convbottleneck_BatchNorm_beta =   MSDNet_Layer1_2[1].modules[0].modules[1].modules[1].bias
Msdnet_b0_step1_scale1_down_convbottleneck_BatchNorm_gamma =  MSDNet_Layer1_2[1].modules[0].modules[1].modules[1].weight
Msdnet_b0_step1_scale1_down_convdowm_weights =                MSDNet_Layer1_2[1].modules[0].modules[1].modules[3].weight
Msdnet_b0_step1_scale1_down_convdowm_BatchNorm_beta =         MSDNet_Layer1_2[1].modules[0].modules[1].modules[4].bias
Msdnet_b0_step1_scale1_down_convdowm_BatchNorm_gamma =        MSDNet_Layer1_2[1].modules[0].modules[1].modules[4].weight

Msdnet_b0_step1_scale1_normal_convbottleneck_weights =        MSDNet_Layer1_2[1].modules[0].modules[2].modules[0].weight
Msdnet_b0_step1_scale1_normal_convbottleneck_BatchNorm_beta = MSDNet_Layer1_2[1].modules[0].modules[2].modules[1].bias
Msdnet_b0_step1_scale1_normal_convbottleneck_BatchNorm_gamma =MSDNet_Layer1_2[1].modules[0].modules[2].modules[1].weight
Msdnet_b0_step1_scale1_normal_convnormal_weights =            MSDNet_Layer1_2[1].modules[0].modules[2].modules[3].weight  #(16, 32, 3, 3)
Msdnet_b0_step1_scale1_normal_convnormal_BatchNorm_beta =     MSDNet_Layer1_2[1].modules[0].modules[2].modules[4].bias
Msdnet_b0_step1_scale1_normal_convnormal_BatchNorm_gamma =    MSDNet_Layer1_2[1].modules[0].modules[2].modules[4].weight

Msdnet_b0_step1_scale2_down_convbottleneck_weights =          MSDNet_Layer1_2[2].modules[0].modules[1].modules[0].weight #(64, 96, 1, 1)
Msdnet_b0_step1_scale2_down_convbottleneck_BatchNorm_beta =   MSDNet_Layer1_2[2].modules[0].modules[1].modules[1].bias
Msdnet_b0_step1_scale2_down_convbottleneck_BatchNorm_gamma =  MSDNet_Layer1_2[2].modules[0].modules[1].modules[1].weight
Msdnet_b0_step1_scale2_down_convdowm_weights =                MSDNet_Layer1_2[2].modules[0].modules[1].modules[3].weight
Msdnet_b0_step1_scale2_down_convdowm_BatchNorm_beta =         MSDNet_Layer1_2[2].modules[0].modules[1].modules[4].bias
Msdnet_b0_step1_scale2_down_convdowm_BatchNorm_gamma =        MSDNet_Layer1_2[2].modules[0].modules[1].modules[4].weight

Msdnet_b0_step1_scale2_normal_convbottleneck_weights =        MSDNet_Layer1_2[2].modules[0].modules[2].modules[0].weight #(128, 192, 1, 1)
Msdnet_b0_step1_scale2_normal_convbottleneck_BatchNorm_beta = MSDNet_Layer1_2[2].modules[0].modules[2].modules[1].bias
Msdnet_b0_step1_scale2_normal_convbottleneck_BatchNorm_gamma =MSDNet_Layer1_2[2].modules[0].modules[2].modules[1].weight
Msdnet_b0_step1_scale2_normal_convnormal_weights =            MSDNet_Layer1_2[2].modules[0].modules[2].modules[3].weight
Msdnet_b0_step1_scale2_normal_convnormal_BatchNorm_beta =     MSDNet_Layer1_2[2].modules[0].modules[2].modules[4].bias
Msdnet_b0_step1_scale2_normal_convnormal_BatchNorm_gamma =    MSDNet_Layer1_2[2].modules[0].modules[2].modules[4].weight

Msdnet_b0_step1_scale3_down_convbottleneck_weights =          MSDNet_Layer1_2[3].modules[0].modules[1].modules[0].weight
Msdnet_b0_step1_scale3_down_convbottleneck_BatchNorm_beta =   MSDNet_Layer1_2[3].modules[0].modules[1].modules[1].bias
Msdnet_b0_step1_scale3_down_convbottleneck_BatchNorm_gamma =  MSDNet_Layer1_2[3].modules[0].modules[1].modules[1].weight
Msdnet_b0_step1_scale3_down_convdowm_weights =                MSDNet_Layer1_2[3].modules[0].modules[1].modules[3].weight
Msdnet_b0_step1_scale3_down_convdowm_BatchNorm_beta =         MSDNet_Layer1_2[3].modules[0].modules[1].modules[4].bias
Msdnet_b0_step1_scale3_down_convdowm_BatchNorm_gamma =        MSDNet_Layer1_2[3].modules[0].modules[1].modules[4].weight

Msdnet_b0_step1_scale3_normal_convbottleneck_weights =        MSDNet_Layer1_2[3].modules[0].modules[2].modules[0].weight
Msdnet_b0_step1_scale3_normal_convbottleneck_BatchNorm_beta = MSDNet_Layer1_2[3].modules[0].modules[2].modules[1].bias
Msdnet_b0_step1_scale3_normal_convbottleneck_BatchNorm_gamma =MSDNet_Layer1_2[3].modules[0].modules[2].modules[1].weight
Msdnet_b0_step1_scale3_normal_convnormal_weights =            MSDNet_Layer1_2[3].modules[0].modules[2].modules[3].weight
Msdnet_b0_step1_scale3_normal_convnormal_BatchNorm_beta =     MSDNet_Layer1_2[3].modules[0].modules[2].modules[4].bias
Msdnet_b0_step1_scale3_normal_convnormal_BatchNorm_gamma =    MSDNet_Layer1_2[3].modules[0].modules[2].modules[4].weight

MSDNet_Layer1_3 = block1[3].modules
Msdnet_b0_step2_scale0_normal_convbottleneck_weights =               MSDNet_Layer1_3[0].modules[0].modules[1].modules[0].weight
Msdnet_b0_step2_scale0_normal_convbottleneck_BatchNorm_beta =        MSDNet_Layer1_3[0].modules[0].modules[1].modules[1].bias
Msdnet_b0_step2_scale0_normal_convbottleneck_BatchNorm_gamma =       MSDNet_Layer1_3[0].modules[0].modules[1].modules[1].weight
Msdnet_b0_step2_scale0_normal_convnormal_weights =                   MSDNet_Layer1_3[0].modules[0].modules[1].modules[3].weight
Msdnet_b0_step2_scale0_normal_convnormal_BatchNorm_beta =            MSDNet_Layer1_3[0].modules[0].modules[1].modules[4].bias
Msdnet_b0_step2_scale0_normal_convnormal_BatchNorm_gamma =           MSDNet_Layer1_3[0].modules[0].modules[1].modules[4].weight
Msdnet_b0_step2_scale1_down_convbottleneck_weights =                 MSDNet_Layer1_3[1].modules[0].modules[1].modules[0].weight
Msdnet_b0_step2_scale1_down_convbottleneck_BatchNorm_beta =          MSDNet_Layer1_3[1].modules[0].modules[1].modules[1].bias
Msdnet_b0_step2_scale1_down_convbottleneck_BatchNorm_gamma =         MSDNet_Layer1_3[1].modules[0].modules[1].modules[1].weight
Msdnet_b0_step2_scale1_down_convdowm_weights =                       MSDNet_Layer1_3[1].modules[0].modules[1].modules[3].weight
Msdnet_b0_step2_scale1_down_convdowm_BatchNorm_beta =                MSDNet_Layer1_3[1].modules[0].modules[1].modules[4].bias
Msdnet_b0_step2_scale1_down_convdowm_BatchNorm_gamma =               MSDNet_Layer1_3[1].modules[0].modules[1].modules[4].weight
Msdnet_b0_step2_scale1_normal_convbottleneck_weights =               MSDNet_Layer1_3[1].modules[0].modules[2].modules[0].weight
Msdnet_b0_step2_scale1_normal_convbottleneck_BatchNorm_beta =        MSDNet_Layer1_3[1].modules[0].modules[2].modules[1].bias
Msdnet_b0_step2_scale1_normal_convbottleneck_BatchNorm_gamma =       MSDNet_Layer1_3[1].modules[0].modules[2].modules[1].weight
Msdnet_b0_step2_scale1_normal_convnormal_weights =                   MSDNet_Layer1_3[1].modules[0].modules[2].modules[3].weight
Msdnet_b0_step2_scale1_normal_convnormal_BatchNorm_beta =            MSDNet_Layer1_3[1].modules[0].modules[2].modules[4].bias
Msdnet_b0_step2_scale1_normal_convnormal_BatchNorm_gamma =           MSDNet_Layer1_3[1].modules[0].modules[2].modules[4].weight
Msdnet_b0_step2_scale2_down_convbottleneck_weights =                 MSDNet_Layer1_3[2].modules[0].modules[1].modules[0].weight
Msdnet_b0_step2_scale2_down_convbottleneck_BatchNorm_beta =          MSDNet_Layer1_3[2].modules[0].modules[1].modules[1].bias
Msdnet_b0_step2_scale2_down_convbottleneck_BatchNorm_gamma =         MSDNet_Layer1_3[2].modules[0].modules[1].modules[1].weight
Msdnet_b0_step2_scale2_down_convdowm_weights =                       MSDNet_Layer1_3[2].modules[0].modules[1].modules[3].weight
Msdnet_b0_step2_scale2_down_convdowm_BatchNorm_beta =                MSDNet_Layer1_3[2].modules[0].modules[1].modules[4].bias  
Msdnet_b0_step2_scale2_down_convdowm_BatchNorm_gamma =               MSDNet_Layer1_3[2].modules[0].modules[1].modules[4].weight
Msdnet_b0_step2_scale2_normal_convbottleneck_weights =               MSDNet_Layer1_3[2].modules[0].modules[2].modules[0].weight
Msdnet_b0_step2_scale2_normal_convbottleneck_BatchNorm_beta =        MSDNet_Layer1_3[2].modules[0].modules[2].modules[1].bias  
Msdnet_b0_step2_scale2_normal_convbottleneck_BatchNorm_gamma =       MSDNet_Layer1_3[2].modules[0].modules[2].modules[1].weight
Msdnet_b0_step2_scale2_normal_convnormal_weights =                   MSDNet_Layer1_3[2].modules[0].modules[2].modules[3].weight
Msdnet_b0_step2_scale2_normal_convnormal_BatchNorm_beta =            MSDNet_Layer1_3[2].modules[0].modules[2].modules[4].bias
Msdnet_b0_step2_scale2_normal_convnormal_BatchNorm_gamma =           MSDNet_Layer1_3[2].modules[0].modules[2].modules[4].weight
Msdnet_b0_step2_scale3_down_convbottleneck_weights =                 MSDNet_Layer1_3[3].modules[0].modules[1].modules[0].weight
Msdnet_b0_step2_scale3_down_convbottleneck_BatchNorm_beta =          MSDNet_Layer1_3[3].modules[0].modules[1].modules[1].bias
Msdnet_b0_step2_scale3_down_convbottleneck_BatchNorm_gamma =         MSDNet_Layer1_3[3].modules[0].modules[1].modules[1].weight
Msdnet_b0_step2_scale3_down_convdowm_weights =                       MSDNet_Layer1_3[3].modules[0].modules[1].modules[3].weight
Msdnet_b0_step2_scale3_down_convdowm_BatchNorm_beta =                MSDNet_Layer1_3[3].modules[0].modules[1].modules[4].bias
Msdnet_b0_step2_scale3_down_convdowm_BatchNorm_gamma =               MSDNet_Layer1_3[3].modules[0].modules[1].modules[4].weight
Msdnet_b0_step2_scale3_normal_convbottleneck_weights =               MSDNet_Layer1_3[3].modules[0].modules[2].modules[0].weight
Msdnet_b0_step2_scale3_normal_convbottleneck_BatchNorm_beta =        MSDNet_Layer1_3[3].modules[0].modules[2].modules[1].bias
Msdnet_b0_step2_scale3_normal_convbottleneck_BatchNorm_gamma =       MSDNet_Layer1_3[3].modules[0].modules[2].modules[1].weight
Msdnet_b0_step2_scale3_normal_convnormal_weights =                   MSDNet_Layer1_3[3].modules[0].modules[2].modules[3].weight
Msdnet_b0_step2_scale3_normal_convnormal_BatchNorm_beta =            MSDNet_Layer1_3[3].modules[0].modules[2].modules[4].bias
Msdnet_b0_step2_scale3_normal_convnormal_BatchNorm_gamma =           MSDNet_Layer1_3[3].modules[0].modules[2].modules[4].weight

MSDNet_Layer1_4 = block1[4].modules
Msdnet_b0_step3_scale0_normal_convbottleneck_weights =               MSDNet_Layer1_4[0].modules[0].modules[1].modules[0].weight
Msdnet_b0_step3_scale0_normal_convbottleneck_BatchNorm_beta =        MSDNet_Layer1_4[0].modules[0].modules[1].modules[1].bias
Msdnet_b0_step3_scale0_normal_convbottleneck_BatchNorm_gamma =       MSDNet_Layer1_4[0].modules[0].modules[1].modules[1].weight
Msdnet_b0_step3_scale0_normal_convnormal_weights =                   MSDNet_Layer1_4[0].modules[0].modules[1].modules[3].weight
Msdnet_b0_step3_scale0_normal_convnormal_BatchNorm_beta =            MSDNet_Layer1_4[0].modules[0].modules[1].modules[4].bias
Msdnet_b0_step3_scale0_normal_convnormal_BatchNorm_gamma =           MSDNet_Layer1_4[0].modules[0].modules[1].modules[4].weight
Msdnet_b0_step3_scale1_down_convbottleneck_weights =                 MSDNet_Layer1_4[1].modules[0].modules[1].modules[0].weight
Msdnet_b0_step3_scale1_down_convbottleneck_BatchNorm_beta =          MSDNet_Layer1_4[1].modules[0].modules[1].modules[1].bias
Msdnet_b0_step3_scale1_down_convbottleneck_BatchNorm_gamma =         MSDNet_Layer1_4[1].modules[0].modules[1].modules[1].weight
Msdnet_b0_step3_scale1_down_convdowm_weights =                       MSDNet_Layer1_4[1].modules[0].modules[1].modules[3].weight
Msdnet_b0_step3_scale1_down_convdowm_BatchNorm_beta =                MSDNet_Layer1_4[1].modules[0].modules[1].modules[4].bias
Msdnet_b0_step3_scale1_down_convdowm_BatchNorm_gamma =               MSDNet_Layer1_4[1].modules[0].modules[1].modules[4].weight
Msdnet_b0_step3_scale1_normal_convbottleneck_weights =               MSDNet_Layer1_4[1].modules[0].modules[2].modules[0].weight
Msdnet_b0_step3_scale1_normal_convbottleneck_BatchNorm_beta =        MSDNet_Layer1_4[1].modules[0].modules[2].modules[1].bias
Msdnet_b0_step3_scale1_normal_convbottleneck_BatchNorm_gamma =       MSDNet_Layer1_4[1].modules[0].modules[2].modules[1].weight
Msdnet_b0_step3_scale1_normal_convnormal_weights =                   MSDNet_Layer1_4[1].modules[0].modules[2].modules[3].weight
Msdnet_b0_step3_scale1_normal_convnormal_BatchNorm_beta =            MSDNet_Layer1_4[1].modules[0].modules[2].modules[4].bias
Msdnet_b0_step3_scale1_normal_convnormal_BatchNorm_gamma =           MSDNet_Layer1_4[1].modules[0].modules[2].modules[4].weight
Msdnet_b0_step3_scale2_down_convbottleneck_weights =                 MSDNet_Layer1_4[2].modules[0].modules[1].modules[0].weight
Msdnet_b0_step3_scale2_down_convbottleneck_BatchNorm_beta =          MSDNet_Layer1_4[2].modules[0].modules[1].modules[1].bias
Msdnet_b0_step3_scale2_down_convbottleneck_BatchNorm_gamma =         MSDNet_Layer1_4[2].modules[0].modules[1].modules[1].weight
Msdnet_b0_step3_scale2_down_convdowm_weights =                       MSDNet_Layer1_4[2].modules[0].modules[1].modules[3].weight
Msdnet_b0_step3_scale2_down_convdowm_BatchNorm_beta =                MSDNet_Layer1_4[2].modules[0].modules[1].modules[4].bias
Msdnet_b0_step3_scale2_down_convdowm_BatchNorm_gamma =               MSDNet_Layer1_4[2].modules[0].modules[1].modules[4].weight
Msdnet_b0_step3_scale2_normal_convbottleneck_weights =               MSDNet_Layer1_4[2].modules[0].modules[2].modules[0].weight
Msdnet_b0_step3_scale2_normal_convbottleneck_BatchNorm_beta =        MSDNet_Layer1_4[2].modules[0].modules[2].modules[1].bias
Msdnet_b0_step3_scale2_normal_convbottleneck_BatchNorm_gamma =       MSDNet_Layer1_4[2].modules[0].modules[2].modules[1].weight
Msdnet_b0_step3_scale2_normal_convnormal_weights =                   MSDNet_Layer1_4[2].modules[0].modules[2].modules[3].weight
Msdnet_b0_step3_scale2_normal_convnormal_BatchNorm_beta =            MSDNet_Layer1_4[2].modules[0].modules[2].modules[4].bias
Msdnet_b0_step3_scale2_normal_convnormal_BatchNorm_gamma =           MSDNet_Layer1_4[2].modules[0].modules[2].modules[4].weight
Msdnet_b0_step3_scale3_down_convbottleneck_weights =                 MSDNet_Layer1_4[3].modules[0].modules[1].modules[0].weight
Msdnet_b0_step3_scale3_down_convbottleneck_BatchNorm_beta =          MSDNet_Layer1_4[3].modules[0].modules[1].modules[1].bias
Msdnet_b0_step3_scale3_down_convbottleneck_BatchNorm_gamma =         MSDNet_Layer1_4[3].modules[0].modules[1].modules[1].weight
Msdnet_b0_step3_scale3_down_convdowm_weights =                       MSDNet_Layer1_4[3].modules[0].modules[1].modules[3].weight
Msdnet_b0_step3_scale3_down_convdowm_BatchNorm_beta =                MSDNet_Layer1_4[3].modules[0].modules[1].modules[4].bias
Msdnet_b0_step3_scale3_down_convdowm_BatchNorm_gamma =               MSDNet_Layer1_4[3].modules[0].modules[1].modules[4].weight
Msdnet_b0_step3_scale3_normal_convbottleneck_weights =               MSDNet_Layer1_4[3].modules[0].modules[2].modules[0].weight
Msdnet_b0_step3_scale3_normal_convbottleneck_BatchNorm_beta =        MSDNet_Layer1_4[3].modules[0].modules[2].modules[1].bias
Msdnet_b0_step3_scale3_normal_convbottleneck_BatchNorm_gamma =       MSDNet_Layer1_4[3].modules[0].modules[2].modules[1].weight
Msdnet_b0_step3_scale3_normal_convnormal_weights =                   MSDNet_Layer1_4[3].modules[0].modules[2].modules[3].weight
Msdnet_b0_step3_scale3_normal_convnormal_BatchNorm_beta =            MSDNet_Layer1_4[3].modules[0].modules[2].modules[4].bias
Msdnet_b0_step3_scale3_normal_convnormal_BatchNorm_gamma =           MSDNet_Layer1_4[3].modules[0].modules[2].modules[4].weight

#block2
block2 = Joint[1].modules
MSDNet_Layer2_1 = block2[0].modules 
Msdnet_b1_step0_scale0_normal_convbottleneck_weights =               MSDNet_Layer2_1[0].modules[0].modules[1].modules[0].weight
Msdnet_b1_step0_scale0_normal_convbottleneck_BatchNorm_beta =        MSDNet_Layer2_1[0].modules[0].modules[1].modules[1].bias
Msdnet_b1_step0_scale0_normal_convbottleneck_BatchNorm_gamma =       MSDNet_Layer2_1[0].modules[0].modules[1].modules[1].weight
Msdnet_b1_step0_scale0_normal_convnormal_weights =                   MSDNet_Layer2_1[0].modules[0].modules[1].modules[3].weight
Msdnet_b1_step0_scale0_normal_convnormal_BatchNorm_beta =            MSDNet_Layer2_1[0].modules[0].modules[1].modules[4].bias
Msdnet_b1_step0_scale0_normal_convnormal_BatchNorm_gamma =           MSDNet_Layer2_1[0].modules[0].modules[1].modules[4].weight
Msdnet_b1_step0_scale1_down_convbottleneck_weights =                 MSDNet_Layer2_1[1].modules[0].modules[1].modules[0].weight
Msdnet_b1_step0_scale1_down_convbottleneck_BatchNorm_beta =          MSDNet_Layer2_1[1].modules[0].modules[1].modules[1].bias
Msdnet_b1_step0_scale1_down_convbottleneck_BatchNorm_gamma =         MSDNet_Layer2_1[1].modules[0].modules[1].modules[1].weight
Msdnet_b1_step0_scale1_down_convdowm_weights =                       MSDNet_Layer2_1[1].modules[0].modules[1].modules[3].weight
Msdnet_b1_step0_scale1_down_convdowm_BatchNorm_beta =                MSDNet_Layer2_1[1].modules[0].modules[1].modules[4].bias
Msdnet_b1_step0_scale1_down_convdowm_BatchNorm_gamma =               MSDNet_Layer2_1[1].modules[0].modules[1].modules[4].weight
Msdnet_b1_step0_scale1_normal_convbottleneck_weights =               MSDNet_Layer2_1[1].modules[0].modules[2].modules[0].weight
Msdnet_b1_step0_scale1_normal_convbottleneck_BatchNorm_beta =        MSDNet_Layer2_1[1].modules[0].modules[2].modules[1].bias
Msdnet_b1_step0_scale1_normal_convbottleneck_BatchNorm_gamma =       MSDNet_Layer2_1[1].modules[0].modules[2].modules[1].weight
Msdnet_b1_step0_scale1_normal_convnormal_weights =                   MSDNet_Layer2_1[1].modules[0].modules[2].modules[3].weight
Msdnet_b1_step0_scale1_normal_convnormal_BatchNorm_beta =            MSDNet_Layer2_1[1].modules[0].modules[2].modules[4].bias
Msdnet_b1_step0_scale1_normal_convnormal_BatchNorm_gamma =           MSDNet_Layer2_1[1].modules[0].modules[2].modules[4].weight
Msdnet_b1_step0_scale2_down_convbottleneck_weights =                 MSDNet_Layer2_1[2].modules[0].modules[1].modules[0].weight
Msdnet_b1_step0_scale2_down_convbottleneck_BatchNorm_beta =          MSDNet_Layer2_1[2].modules[0].modules[1].modules[1].bias
Msdnet_b1_step0_scale2_down_convbottleneck_BatchNorm_gamma =         MSDNet_Layer2_1[2].modules[0].modules[1].modules[1].weight
Msdnet_b1_step0_scale2_down_convdowm_weights =                       MSDNet_Layer2_1[2].modules[0].modules[1].modules[3].weight
Msdnet_b1_step0_scale2_down_convdowm_BatchNorm_beta =                MSDNet_Layer2_1[2].modules[0].modules[1].modules[4].bias
Msdnet_b1_step0_scale2_down_convdowm_BatchNorm_gamma =               MSDNet_Layer2_1[2].modules[0].modules[1].modules[4].weight
Msdnet_b1_step0_scale2_normal_convbottleneck_weights =               MSDNet_Layer2_1[2].modules[0].modules[2].modules[0].weight
Msdnet_b1_step0_scale2_normal_convbottleneck_BatchNorm_beta =        MSDNet_Layer2_1[2].modules[0].modules[2].modules[1].bias
Msdnet_b1_step0_scale2_normal_convbottleneck_BatchNorm_gamma =       MSDNet_Layer2_1[2].modules[0].modules[2].modules[1].weight
Msdnet_b1_step0_scale2_normal_convnormal_weights =                   MSDNet_Layer2_1[2].modules[0].modules[2].modules[3].weight
Msdnet_b1_step0_scale2_normal_convnormal_BatchNorm_beta =            MSDNet_Layer2_1[2].modules[0].modules[2].modules[4].bias
Msdnet_b1_step0_scale2_normal_convnormal_BatchNorm_gamma =           MSDNet_Layer2_1[2].modules[0].modules[2].modules[4].weight
Msdnet_b1_step0_scale3_down_convbottleneck_weights =                 MSDNet_Layer2_1[3].modules[0].modules[1].modules[0].weight
Msdnet_b1_step0_scale3_down_convbottleneck_BatchNorm_beta =          MSDNet_Layer2_1[3].modules[0].modules[1].modules[1].bias
Msdnet_b1_step0_scale3_down_convbottleneck_BatchNorm_gamma =         MSDNet_Layer2_1[3].modules[0].modules[1].modules[1].weight
Msdnet_b1_step0_scale3_down_convdowm_weights =                       MSDNet_Layer2_1[3].modules[0].modules[1].modules[3].weight
Msdnet_b1_step0_scale3_down_convdowm_BatchNorm_beta =                MSDNet_Layer2_1[3].modules[0].modules[1].modules[4].bias
Msdnet_b1_step0_scale3_down_convdowm_BatchNorm_gamma =               MSDNet_Layer2_1[3].modules[0].modules[1].modules[4].weight
Msdnet_b1_step0_scale3_normal_convbottleneck_weights =               MSDNet_Layer2_1[3].modules[0].modules[2].modules[0].weight
Msdnet_b1_step0_scale3_normal_convbottleneck_BatchNorm_beta =        MSDNet_Layer2_1[3].modules[0].modules[2].modules[1].bias
Msdnet_b1_step0_scale3_normal_convbottleneck_BatchNorm_gamma =       MSDNet_Layer2_1[3].modules[0].modules[2].modules[1].weight
Msdnet_b1_step0_scale3_normal_convnormal_weights =                   MSDNet_Layer2_1[3].modules[0].modules[2].modules[3].weight
Msdnet_b1_step0_scale3_normal_convnormal_BatchNorm_beta =            MSDNet_Layer2_1[3].modules[0].modules[2].modules[4].bias
Msdnet_b1_step0_scale3_normal_convnormal_BatchNorm_gamma =           MSDNet_Layer2_1[3].modules[0].modules[2].modules[4].weight

MSDNet_Layer2_2 = block2[1].modules
Msdnet_b1_step1_scale0_down_convbottleneck_weights =                 MSDNet_Layer2_2[0].modules[0].modules[1].modules[0].weight
Msdnet_b1_step1_scale0_down_convbottleneck_BatchNorm_beta =          MSDNet_Layer2_2[0].modules[0].modules[1].modules[1].bias
Msdnet_b1_step1_scale0_down_convbottleneck_BatchNorm_gamma =         MSDNet_Layer2_2[0].modules[0].modules[1].modules[1].weight
Msdnet_b1_step1_scale0_down_convdowm_weights =                       MSDNet_Layer2_2[0].modules[0].modules[1].modules[3].weight
Msdnet_b1_step1_scale0_down_convdowm_BatchNorm_beta =                MSDNet_Layer2_2[0].modules[0].modules[1].modules[4].bias
Msdnet_b1_step1_scale0_down_convdowm_BatchNorm_gamma =               MSDNet_Layer2_2[0].modules[0].modules[1].modules[4].weight
Msdnet_b1_step1_scale0_normal_convbottleneck_weights =               MSDNet_Layer2_2[0].modules[0].modules[2].modules[0].weight
Msdnet_b1_step1_scale0_normal_convbottleneck_BatchNorm_beta =        MSDNet_Layer2_2[0].modules[0].modules[2].modules[1].bias
Msdnet_b1_step1_scale0_normal_convbottleneck_BatchNorm_gamma =       MSDNet_Layer2_2[0].modules[0].modules[2].modules[1].weight
Msdnet_b1_step1_scale0_normal_convnormal_weights =                   MSDNet_Layer2_2[0].modules[0].modules[2].modules[3].weight
Msdnet_b1_step1_scale0_normal_convnormal_BatchNorm_beta =            MSDNet_Layer2_2[0].modules[0].modules[2].modules[4].bias
Msdnet_b1_step1_scale0_normal_convnormal_BatchNorm_gamma =           MSDNet_Layer2_2[0].modules[0].modules[2].modules[4].weight
Msdnet_b1_step1_scale1_down_convbottleneck_weights =                 MSDNet_Layer2_2[1].modules[0].modules[1].modules[0].weight
Msdnet_b1_step1_scale1_down_convbottleneck_BatchNorm_beta =          MSDNet_Layer2_2[1].modules[0].modules[1].modules[1].bias
Msdnet_b1_step1_scale1_down_convbottleneck_BatchNorm_gamma =         MSDNet_Layer2_2[1].modules[0].modules[1].modules[1].weight
Msdnet_b1_step1_scale1_down_convdowm_weights =                       MSDNet_Layer2_2[1].modules[0].modules[1].modules[3].weight
Msdnet_b1_step1_scale1_down_convdowm_BatchNorm_beta =                MSDNet_Layer2_2[1].modules[0].modules[1].modules[4].bias
Msdnet_b1_step1_scale1_down_convdowm_BatchNorm_gamma =               MSDNet_Layer2_2[1].modules[0].modules[1].modules[4].weight
Msdnet_b1_step1_scale1_normal_convbottleneck_weights =               MSDNet_Layer2_2[1].modules[0].modules[2].modules[0].weight
Msdnet_b1_step1_scale1_normal_convbottleneck_BatchNorm_beta =        MSDNet_Layer2_2[1].modules[0].modules[2].modules[1].bias
Msdnet_b1_step1_scale1_normal_convbottleneck_BatchNorm_gamma =       MSDNet_Layer2_2[1].modules[0].modules[2].modules[1].weight
Msdnet_b1_step1_scale1_normal_convnormal_weights =                   MSDNet_Layer2_2[1].modules[0].modules[2].modules[3].weight
Msdnet_b1_step1_scale1_normal_convnormal_BatchNorm_beta =            MSDNet_Layer2_2[1].modules[0].modules[2].modules[4].bias
Msdnet_b1_step1_scale1_normal_convnormal_BatchNorm_gamma =           MSDNet_Layer2_2[1].modules[0].modules[2].modules[4].weight
Msdnet_b1_step1_scale2_down_convbottleneck_weights =                 MSDNet_Layer2_2[2].modules[0].modules[1].modules[0].weight
Msdnet_b1_step1_scale2_down_convbottleneck_BatchNorm_beta =          MSDNet_Layer2_2[2].modules[0].modules[1].modules[1].bias
Msdnet_b1_step1_scale2_down_convbottleneck_BatchNorm_gamma =         MSDNet_Layer2_2[2].modules[0].modules[1].modules[1].weight
Msdnet_b1_step1_scale2_down_convdowm_weights =                       MSDNet_Layer2_2[2].modules[0].modules[1].modules[3].weight
Msdnet_b1_step1_scale2_down_convdowm_BatchNorm_beta =                MSDNet_Layer2_2[2].modules[0].modules[1].modules[4].bias
Msdnet_b1_step1_scale2_down_convdowm_BatchNorm_gamma =               MSDNet_Layer2_2[2].modules[0].modules[1].modules[4].weight
Msdnet_b1_step1_scale2_normal_convbottleneck_weights =               MSDNet_Layer2_2[2].modules[0].modules[2].modules[0].weight
Msdnet_b1_step1_scale2_normal_convbottleneck_BatchNorm_beta =        MSDNet_Layer2_2[2].modules[0].modules[2].modules[1].bias
Msdnet_b1_step1_scale2_normal_convbottleneck_BatchNorm_gamma =       MSDNet_Layer2_2[2].modules[0].modules[2].modules[1].weight
Msdnet_b1_step1_scale2_normal_convnormal_weights =                   MSDNet_Layer2_2[2].modules[0].modules[2].modules[3].weight
Msdnet_b1_step1_scale2_normal_convnormal_BatchNorm_beta =            MSDNet_Layer2_2[2].modules[0].modules[2].modules[4].bias
Msdnet_b1_step1_scale2_normal_convnormal_BatchNorm_gamma =           MSDNet_Layer2_2[2].modules[0].modules[2].modules[4].weight

Trans2 = block2[2].modules
Msdnet_b1_transition_scale0_conv1_weights = Trans2[0].modules[0].weight    #(128, 256, 1, 1)
Msdnet_b1_transition_scale0_conv1_BatchNorm_beta =Trans2[0].modules[1].bias
Msdnet_b1_transition_scale0_conv1_BatchNorm_gamma =Trans2[0].modules[1].weight
Msdnet_b1_transition_scale1_conv1_weights =Trans2[1].modules[0].weight
Msdnet_b1_transition_scale1_conv1_BatchNorm_beta =Trans2[1].modules[1].bias
Msdnet_b1_transition_scale1_conv1_BatchNorm_gamma =Trans2[1].modules[1].weight
Msdnet_b1_transition_scale2_conv1_weights =Trans2[2].modules[0].weight
Msdnet_b1_transition_scale2_conv1_BatchNorm_beta =Trans2[2].modules[1].bias
Msdnet_b1_transition_scale2_conv1_BatchNorm_gamma =Trans2[2].modules[1].weight

MSDNet_Layer2_3 = block2[3].modules
Msdnet_b1_step2_scale0_normal_convbottleneck_weights =            MSDNet_Layer2_3[0].modules[0].modules[1].modules[0].weight
Msdnet_b1_step2_scale0_normal_convbottleneck_BatchNorm_beta =     MSDNet_Layer2_3[0].modules[0].modules[1].modules[1].bias
Msdnet_b1_step2_scale0_normal_convbottleneck_BatchNorm_gamma =    MSDNet_Layer2_3[0].modules[0].modules[1].modules[1].weight
Msdnet_b1_step2_scale0_normal_convnormal_weights =                MSDNet_Layer2_3[0].modules[0].modules[1].modules[3].weight
Msdnet_b1_step2_scale0_normal_convnormal_BatchNorm_beta =         MSDNet_Layer2_3[0].modules[0].modules[1].modules[4].bias
Msdnet_b1_step2_scale0_normal_convnormal_BatchNorm_gamma =        MSDNet_Layer2_3[0].modules[0].modules[1].modules[4].weight
Msdnet_b1_step2_scale1_down_convbottleneck_weights =              MSDNet_Layer2_3[1].modules[0].modules[1].modules[0].weight
Msdnet_b1_step2_scale1_down_convbottleneck_BatchNorm_beta =       MSDNet_Layer2_3[1].modules[0].modules[1].modules[1].bias
Msdnet_b1_step2_scale1_down_convbottleneck_BatchNorm_gamma =      MSDNet_Layer2_3[1].modules[0].modules[1].modules[1].weight
Msdnet_b1_step2_scale1_down_convdowm_weights =                    MSDNet_Layer2_3[1].modules[0].modules[1].modules[3].weight
Msdnet_b1_step2_scale1_down_convdowm_BatchNorm_beta =             MSDNet_Layer2_3[1].modules[0].modules[1].modules[4].bias
Msdnet_b1_step2_scale1_down_convdowm_BatchNorm_gamma =            MSDNet_Layer2_3[1].modules[0].modules[1].modules[4].weight
Msdnet_b1_step2_scale1_normal_convbottleneck_weights =            MSDNet_Layer2_3[1].modules[0].modules[2].modules[0].weight
Msdnet_b1_step2_scale1_normal_convbottleneck_BatchNorm_beta =     MSDNet_Layer2_3[1].modules[0].modules[2].modules[1].bias
Msdnet_b1_step2_scale1_normal_convbottleneck_BatchNorm_gamma =    MSDNet_Layer2_3[1].modules[0].modules[2].modules[1].weight
Msdnet_b1_step2_scale1_normal_convnormal_weights =                MSDNet_Layer2_3[1].modules[0].modules[2].modules[3].weight
Msdnet_b1_step2_scale1_normal_convnormal_BatchNorm_beta =         MSDNet_Layer2_3[1].modules[0].modules[2].modules[4].bias
Msdnet_b1_step2_scale1_normal_convnormal_BatchNorm_gamma =        MSDNet_Layer2_3[1].modules[0].modules[2].modules[4].weight
Msdnet_b1_step2_scale2_down_convbottleneck_weights =              MSDNet_Layer2_3[2].modules[0].modules[1].modules[0].weight
Msdnet_b1_step2_scale2_down_convbottleneck_BatchNorm_beta =       MSDNet_Layer2_3[2].modules[0].modules[1].modules[1].bias
Msdnet_b1_step2_scale2_down_convbottleneck_BatchNorm_gamma =      MSDNet_Layer2_3[2].modules[0].modules[1].modules[1].weight
Msdnet_b1_step2_scale2_down_convdowm_weights =                    MSDNet_Layer2_3[2].modules[0].modules[1].modules[3].weight
Msdnet_b1_step2_scale2_down_convdowm_BatchNorm_beta =             MSDNet_Layer2_3[2].modules[0].modules[1].modules[4].bias
Msdnet_b1_step2_scale2_down_convdowm_BatchNorm_gamma =            MSDNet_Layer2_3[2].modules[0].modules[1].modules[4].weight
Msdnet_b1_step2_scale2_normal_convbottleneck_weights =            MSDNet_Layer2_3[2].modules[0].modules[2].modules[0].weight
Msdnet_b1_step2_scale2_normal_convbottleneck_BatchNorm_beta =     MSDNet_Layer2_3[2].modules[0].modules[2].modules[1].bias
Msdnet_b1_step2_scale2_normal_convbottleneck_BatchNorm_gamma =    MSDNet_Layer2_3[2].modules[0].modules[2].modules[1].weight
Msdnet_b1_step2_scale2_normal_convnormal_weights =                MSDNet_Layer2_3[2].modules[0].modules[2].modules[3].weight
Msdnet_b1_step2_scale2_normal_convnormal_BatchNorm_beta =         MSDNet_Layer2_3[2].modules[0].modules[2].modules[4].bias
Msdnet_b1_step2_scale2_normal_convnormal_BatchNorm_gamma =        MSDNet_Layer2_3[2].modules[0].modules[2].modules[4].weight

MSDNet_Layer2_4 = block2[4].modules
Msdnet_b1_step3_scale0_normal_convbottleneck_weights =            MSDNet_Layer2_4[0].modules[0].modules[1].modules[0].weight
Msdnet_b1_step3_scale0_normal_convbottleneck_BatchNorm_beta =     MSDNet_Layer2_4[0].modules[0].modules[1].modules[1].bias
Msdnet_b1_step3_scale0_normal_convbottleneck_BatchNorm_gamma =    MSDNet_Layer2_4[0].modules[0].modules[1].modules[1].weight
Msdnet_b1_step3_scale0_normal_convnormal_weights =                MSDNet_Layer2_4[0].modules[0].modules[1].modules[3].weight
Msdnet_b1_step3_scale0_normal_convnormal_BatchNorm_beta =         MSDNet_Layer2_4[0].modules[0].modules[1].modules[4].bias
Msdnet_b1_step3_scale0_normal_convnormal_BatchNorm_gamma =        MSDNet_Layer2_4[0].modules[0].modules[1].modules[4].weight
Msdnet_b1_step3_scale1_down_convbottleneck_weights =              MSDNet_Layer2_4[1].modules[0].modules[1].modules[0].weight
Msdnet_b1_step3_scale1_down_convbottleneck_BatchNorm_beta =       MSDNet_Layer2_4[1].modules[0].modules[1].modules[1].bias
Msdnet_b1_step3_scale1_down_convbottleneck_BatchNorm_gamma =      MSDNet_Layer2_4[1].modules[0].modules[1].modules[1].weight
Msdnet_b1_step3_scale1_down_convdowm_weights =                    MSDNet_Layer2_4[1].modules[0].modules[1].modules[3].weight
Msdnet_b1_step3_scale1_down_convdowm_BatchNorm_beta =             MSDNet_Layer2_4[1].modules[0].modules[1].modules[4].bias
Msdnet_b1_step3_scale1_down_convdowm_BatchNorm_gamma =            MSDNet_Layer2_4[1].modules[0].modules[1].modules[4].weight
Msdnet_b1_step3_scale1_normal_convbottleneck_weights =            MSDNet_Layer2_4[1].modules[0].modules[2].modules[0].weight
Msdnet_b1_step3_scale1_normal_convbottleneck_BatchNorm_beta =     MSDNet_Layer2_4[1].modules[0].modules[2].modules[1].bias
Msdnet_b1_step3_scale1_normal_convbottleneck_BatchNorm_gamma =    MSDNet_Layer2_4[1].modules[0].modules[2].modules[1].weight
Msdnet_b1_step3_scale1_normal_convnormal_weights =                MSDNet_Layer2_4[1].modules[0].modules[2].modules[3].weight
Msdnet_b1_step3_scale1_normal_convnormal_BatchNorm_beta =         MSDNet_Layer2_4[1].modules[0].modules[2].modules[4].bias
Msdnet_b1_step3_scale1_normal_convnormal_BatchNorm_gamma =        MSDNet_Layer2_4[1].modules[0].modules[2].modules[4].weight
Msdnet_b1_step3_scale2_down_convbottleneck_weights =              MSDNet_Layer2_4[2].modules[0].modules[1].modules[0].weight
Msdnet_b1_step3_scale2_down_convbottleneck_BatchNorm_beta =       MSDNet_Layer2_4[2].modules[0].modules[1].modules[1].bias
Msdnet_b1_step3_scale2_down_convbottleneck_BatchNorm_gamma =      MSDNet_Layer2_4[2].modules[0].modules[1].modules[1].weight
Msdnet_b1_step3_scale2_down_convdowm_weights =                    MSDNet_Layer2_4[2].modules[0].modules[1].modules[3].weight
Msdnet_b1_step3_scale2_down_convdowm_BatchNorm_beta =             MSDNet_Layer2_4[2].modules[0].modules[1].modules[4].bias
Msdnet_b1_step3_scale2_down_convdowm_BatchNorm_gamma =            MSDNet_Layer2_4[2].modules[0].modules[1].modules[4].weight
Msdnet_b1_step3_scale2_normal_convbottleneck_weights =            MSDNet_Layer2_4[2].modules[0].modules[2].modules[0].weight
Msdnet_b1_step3_scale2_normal_convbottleneck_BatchNorm_beta =     MSDNet_Layer2_4[2].modules[0].modules[2].modules[1].bias
Msdnet_b1_step3_scale2_normal_convbottleneck_BatchNorm_gamma =    MSDNet_Layer2_4[2].modules[0].modules[2].modules[1].weight
Msdnet_b1_step3_scale2_normal_convnormal_weights =                MSDNet_Layer2_4[2].modules[0].modules[2].modules[3].weight
Msdnet_b1_step3_scale2_normal_convnormal_BatchNorm_beta =         MSDNet_Layer2_4[2].modules[0].modules[2].modules[4].bias
Msdnet_b1_step3_scale2_normal_convnormal_BatchNorm_gamma =        MSDNet_Layer2_4[2].modules[0].modules[2].modules[4].weight
#block3
block3 = Joint[2].modules
MSDNet_Layer3_1 = block3[0].modules
Msdnet_b2_step0_scale0_normal_convbottleneck_weights =            MSDNet_Layer3_1[0].modules[0].modules[1].modules[0].weight
Msdnet_b2_step0_scale0_normal_convbottleneck_BatchNorm_beta =     MSDNet_Layer3_1[0].modules[0].modules[1].modules[1].bias
Msdnet_b2_step0_scale0_normal_convbottleneck_BatchNorm_gamma =    MSDNet_Layer3_1[0].modules[0].modules[1].modules[1].weight
Msdnet_b2_step0_scale0_normal_convnormal_weights =                MSDNet_Layer3_1[0].modules[0].modules[1].modules[3].weight
Msdnet_b2_step0_scale0_normal_convnormal_BatchNorm_beta =         MSDNet_Layer3_1[0].modules[0].modules[1].modules[4].bias
Msdnet_b2_step0_scale0_normal_convnormal_BatchNorm_gamma =        MSDNet_Layer3_1[0].modules[0].modules[1].modules[4].weight
Msdnet_b2_step0_scale1_down_convbottleneck_weights =              MSDNet_Layer3_1[1].modules[0].modules[1].modules[0].weight
Msdnet_b2_step0_scale1_down_convbottleneck_BatchNorm_beta =       MSDNet_Layer3_1[1].modules[0].modules[1].modules[1].bias
Msdnet_b2_step0_scale1_down_convbottleneck_BatchNorm_gamma =      MSDNet_Layer3_1[1].modules[0].modules[1].modules[1].weight
Msdnet_b2_step0_scale1_down_convdowm_weights =                    MSDNet_Layer3_1[1].modules[0].modules[1].modules[3].weight
Msdnet_b2_step0_scale1_down_convdowm_BatchNorm_beta =             MSDNet_Layer3_1[1].modules[0].modules[1].modules[4].bias
Msdnet_b2_step0_scale1_down_convdowm_BatchNorm_gamma =            MSDNet_Layer3_1[1].modules[0].modules[1].modules[4].weight
Msdnet_b2_step0_scale1_normal_convbottleneck_weights =            MSDNet_Layer3_1[1].modules[0].modules[2].modules[0].weight
Msdnet_b2_step0_scale1_normal_convbottleneck_BatchNorm_beta =     MSDNet_Layer3_1[1].modules[0].modules[2].modules[1].bias
Msdnet_b2_step0_scale1_normal_convbottleneck_BatchNorm_gamma =    MSDNet_Layer3_1[1].modules[0].modules[2].modules[1].weight
Msdnet_b2_step0_scale1_normal_convnormal_weights =                MSDNet_Layer3_1[1].modules[0].modules[2].modules[3].weight
Msdnet_b2_step0_scale1_normal_convnormal_BatchNorm_beta =         MSDNet_Layer3_1[1].modules[0].modules[2].modules[4].bias
Msdnet_b2_step0_scale1_normal_convnormal_BatchNorm_gamma =        MSDNet_Layer3_1[1].modules[0].modules[2].modules[4].weight
Msdnet_b2_step0_scale2_down_convbottleneck_weights =              MSDNet_Layer3_1[2].modules[0].modules[1].modules[0].weight
Msdnet_b2_step0_scale2_down_convbottleneck_BatchNorm_beta =       MSDNet_Layer3_1[2].modules[0].modules[1].modules[1].bias
Msdnet_b2_step0_scale2_down_convbottleneck_BatchNorm_gamma =      MSDNet_Layer3_1[2].modules[0].modules[1].modules[1].weight
Msdnet_b2_step0_scale2_down_convdowm_weights =                    MSDNet_Layer3_1[2].modules[0].modules[1].modules[3].weight
Msdnet_b2_step0_scale2_down_convdowm_BatchNorm_beta =             MSDNet_Layer3_1[2].modules[0].modules[1].modules[4].bias
Msdnet_b2_step0_scale2_down_convdowm_BatchNorm_gamma =            MSDNet_Layer3_1[2].modules[0].modules[1].modules[4].weight
Msdnet_b2_step0_scale2_normal_convbottleneck_weights =            MSDNet_Layer3_1[2].modules[0].modules[2].modules[0].weight
Msdnet_b2_step0_scale2_normal_convbottleneck_BatchNorm_beta =     MSDNet_Layer3_1[2].modules[0].modules[2].modules[1].bias
Msdnet_b2_step0_scale2_normal_convbottleneck_BatchNorm_gamma =    MSDNet_Layer3_1[2].modules[0].modules[2].modules[1].weight
Msdnet_b2_step0_scale2_normal_convnormal_weights =                MSDNet_Layer3_1[2].modules[0].modules[2].modules[3].weight
Msdnet_b2_step0_scale2_normal_convnormal_BatchNorm_beta =         MSDNet_Layer3_1[2].modules[0].modules[2].modules[4].bias
Msdnet_b2_step0_scale2_normal_convnormal_BatchNorm_gamma =        MSDNet_Layer3_1[2].modules[0].modules[2].modules[4].weight

MSDNet_Layer3_2 = block3[1].modules
Msdnet_b2_step1_scale0_normal_convbottleneck_weights =            MSDNet_Layer3_2[0].modules[0].modules[1].modules[0].weight
Msdnet_b2_step1_scale0_normal_convbottleneck_BatchNorm_beta =     MSDNet_Layer3_2[0].modules[0].modules[1].modules[1].bias
Msdnet_b2_step1_scale0_normal_convbottleneck_BatchNorm_gamma =    MSDNet_Layer3_2[0].modules[0].modules[1].modules[1].weight
Msdnet_b2_step1_scale0_normal_convnormal_weights =                MSDNet_Layer3_2[0].modules[0].modules[1].modules[3].weight
Msdnet_b2_step1_scale0_normal_convnormal_BatchNorm_beta =         MSDNet_Layer3_2[0].modules[0].modules[1].modules[4].bias
Msdnet_b2_step1_scale0_normal_convnormal_BatchNorm_gamma =        MSDNet_Layer3_2[0].modules[0].modules[1].modules[4].weight
Msdnet_b2_step1_scale1_down_convbottleneck_weights =              MSDNet_Layer3_2[1].modules[0].modules[1].modules[0].weight
Msdnet_b2_step1_scale1_down_convbottleneck_BatchNorm_beta =       MSDNet_Layer3_2[1].modules[0].modules[1].modules[1].bias
Msdnet_b2_step1_scale1_down_convbottleneck_BatchNorm_gamma =      MSDNet_Layer3_2[1].modules[0].modules[1].modules[1].weight
Msdnet_b2_step1_scale1_down_convdowm_weights =                    MSDNet_Layer3_2[1].modules[0].modules[1].modules[3].weight
Msdnet_b2_step1_scale1_down_convdowm_BatchNorm_beta =             MSDNet_Layer3_2[1].modules[0].modules[1].modules[4].bias
Msdnet_b2_step1_scale1_down_convdowm_BatchNorm_gamma =            MSDNet_Layer3_2[1].modules[0].modules[1].modules[4].weight
Msdnet_b2_step1_scale1_normal_convbottleneck_weights =            MSDNet_Layer3_2[1].modules[0].modules[2].modules[0].weight
Msdnet_b2_step1_scale1_normal_convbottleneck_BatchNorm_beta =     MSDNet_Layer3_2[1].modules[0].modules[2].modules[1].bias
Msdnet_b2_step1_scale1_normal_convbottleneck_BatchNorm_gamma =    MSDNet_Layer3_2[1].modules[0].modules[2].modules[1].weight
Msdnet_b2_step1_scale1_normal_convnormal_weights =                MSDNet_Layer3_2[1].modules[0].modules[2].modules[3].weight
Msdnet_b2_step1_scale1_normal_convnormal_BatchNorm_beta =         MSDNet_Layer3_2[1].modules[0].modules[2].modules[4].bias
Msdnet_b2_step1_scale1_normal_convnormal_BatchNorm_gamma =        MSDNet_Layer3_2[1].modules[0].modules[2].modules[4].weight
Msdnet_b2_step1_scale2_down_convbottleneck_weights =              MSDNet_Layer3_2[2].modules[0].modules[1].modules[0].weight
Msdnet_b2_step1_scale2_down_convbottleneck_BatchNorm_beta =       MSDNet_Layer3_2[2].modules[0].modules[1].modules[1].bias
Msdnet_b2_step1_scale2_down_convbottleneck_BatchNorm_gamma =      MSDNet_Layer3_2[2].modules[0].modules[1].modules[1].weight
Msdnet_b2_step1_scale2_down_convdowm_weights =                    MSDNet_Layer3_2[2].modules[0].modules[1].modules[3].weight
Msdnet_b2_step1_scale2_down_convdowm_BatchNorm_beta =             MSDNet_Layer3_2[2].modules[0].modules[1].modules[4].bias
Msdnet_b2_step1_scale2_down_convdowm_BatchNorm_gamma =            MSDNet_Layer3_2[2].modules[0].modules[1].modules[4].weight
Msdnet_b2_step1_scale2_normal_convbottleneck_weights =            MSDNet_Layer3_2[2].modules[0].modules[2].modules[0].weight
Msdnet_b2_step1_scale2_normal_convbottleneck_BatchNorm_beta =     MSDNet_Layer3_2[2].modules[0].modules[2].modules[1].bias
Msdnet_b2_step1_scale2_normal_convbottleneck_BatchNorm_gamma =    MSDNet_Layer3_2[2].modules[0].modules[2].modules[1].weight
Msdnet_b2_step1_scale2_normal_convnormal_weights =                MSDNet_Layer3_2[2].modules[0].modules[2].modules[3].weight
Msdnet_b2_step1_scale2_normal_convnormal_BatchNorm_beta =         MSDNet_Layer3_2[2].modules[0].modules[2].modules[4].bias
Msdnet_b2_step1_scale2_normal_convnormal_BatchNorm_gamma =        MSDNet_Layer3_2[2].modules[0].modules[2].modules[4].weight

MSDNet_Layer3_3 = block3[2].modules
Msdnet_b2_step2_scale0_down_convbottleneck_weights =              MSDNet_Layer3_3[0].modules[0].modules[1].modules[0].weight
Msdnet_b2_step2_scale0_down_convbottleneck_BatchNorm_beta =       MSDNet_Layer3_3[0].modules[0].modules[1].modules[1].bias
Msdnet_b2_step2_scale0_down_convbottleneck_BatchNorm_gamma =      MSDNet_Layer3_3[0].modules[0].modules[1].modules[1].weight
Msdnet_b2_step2_scale0_down_convdowm_weights =                    MSDNet_Layer3_3[0].modules[0].modules[1].modules[3].weight
Msdnet_b2_step2_scale0_down_convdowm_BatchNorm_beta =             MSDNet_Layer3_3[0].modules[0].modules[1].modules[4].bias
Msdnet_b2_step2_scale0_down_convdowm_BatchNorm_gamma =            MSDNet_Layer3_3[0].modules[0].modules[1].modules[4].weight
Msdnet_b2_step2_scale0_normal_convbottleneck_weights =            MSDNet_Layer3_3[0].modules[0].modules[2].modules[0].weight
Msdnet_b2_step2_scale0_normal_convbottleneck_BatchNorm_beta =     MSDNet_Layer3_3[0].modules[0].modules[2].modules[1].bias
Msdnet_b2_step2_scale0_normal_convbottleneck_BatchNorm_gamma =    MSDNet_Layer3_3[0].modules[0].modules[2].modules[1].weight
Msdnet_b2_step2_scale0_normal_convnormal_weights =                MSDNet_Layer3_3[0].modules[0].modules[2].modules[3].weight
Msdnet_b2_step2_scale0_normal_convnormal_BatchNorm_beta =         MSDNet_Layer3_3[0].modules[0].modules[2].modules[4].bias
Msdnet_b2_step2_scale0_normal_convnormal_BatchNorm_gamma =        MSDNet_Layer3_3[0].modules[0].modules[2].modules[4].weight
Msdnet_b2_step2_scale1_down_convbottleneck_weights =              MSDNet_Layer3_3[1].modules[0].modules[1].modules[0].weight
Msdnet_b2_step2_scale1_down_convbottleneck_BatchNorm_beta =       MSDNet_Layer3_3[1].modules[0].modules[1].modules[1].bias
Msdnet_b2_step2_scale1_down_convbottleneck_BatchNorm_gamma =      MSDNet_Layer3_3[1].modules[0].modules[1].modules[1].weight
Msdnet_b2_step2_scale1_down_convdowm_weights =                    MSDNet_Layer3_3[1].modules[0].modules[1].modules[3].weight
Msdnet_b2_step2_scale1_down_convdowm_BatchNorm_beta =             MSDNet_Layer3_3[1].modules[0].modules[1].modules[4].bias
Msdnet_b2_step2_scale1_down_convdowm_BatchNorm_gamma =            MSDNet_Layer3_3[1].modules[0].modules[1].modules[4].weight
Msdnet_b2_step2_scale1_normal_convbottleneck_weights =            MSDNet_Layer3_3[1].modules[0].modules[2].modules[0].weight
Msdnet_b2_step2_scale1_normal_convbottleneck_BatchNorm_beta =     MSDNet_Layer3_3[1].modules[0].modules[2].modules[1].bias
Msdnet_b2_step2_scale1_normal_convbottleneck_BatchNorm_gamma =    MSDNet_Layer3_3[1].modules[0].modules[2].modules[1].weight
Msdnet_b2_step2_scale1_normal_convnormal_weights =                MSDNet_Layer3_3[1].modules[0].modules[2].modules[3].weight
Msdnet_b2_step2_scale1_normal_convnormal_BatchNorm_beta =         MSDNet_Layer3_3[1].modules[0].modules[2].modules[4].bias
Msdnet_b2_step2_scale1_normal_convnormal_BatchNorm_gamma =        MSDNet_Layer3_3[1].modules[0].modules[2].modules[4].weight

Trans3 = block3[3].modules
Msdnet_b2_transition_scale0_conv1_weights = Trans3[0].modules[0].weight
Msdnet_b2_transition_scale0_conv1_BatchNorm_beta =  Trans3[0].modules[1].bias
Msdnet_b2_transition_scale0_conv1_BatchNorm_gamma = Trans3[0].modules[1].weight
Msdnet_b2_transition_scale1_conv1_weights = Trans3[1].modules[0].weight
Msdnet_b2_transition_scale1_conv1_BatchNorm_beta =Trans3[1].modules[1].bias
Msdnet_b2_transition_scale1_conv1_BatchNorm_gamma =Trans3[1].modules[1].weight

MSDNet_Layer3_4 = block3[4].modules
Msdnet_b2_step3_scale0_normal_convbottleneck_weights =           MSDNet_Layer3_4[0].modules[0].modules[1].modules[0].weight
Msdnet_b2_step3_scale0_normal_convbottleneck_BatchNorm_beta =    MSDNet_Layer3_4[0].modules[0].modules[1].modules[1].bias
Msdnet_b2_step3_scale0_normal_convbottleneck_BatchNorm_gamma =   MSDNet_Layer3_4[0].modules[0].modules[1].modules[1].weight
Msdnet_b2_step3_scale0_normal_convnormal_weights =               MSDNet_Layer3_4[0].modules[0].modules[1].modules[3].weight
Msdnet_b2_step3_scale0_normal_convnormal_BatchNorm_beta =        MSDNet_Layer3_4[0].modules[0].modules[1].modules[4].bias
Msdnet_b2_step3_scale0_normal_convnormal_BatchNorm_gamma =       MSDNet_Layer3_4[0].modules[0].modules[1].modules[4].weight
Msdnet_b2_step3_scale1_down_convbottleneck_weights =             MSDNet_Layer3_4[1].modules[0].modules[1].modules[0].weight
Msdnet_b2_step3_scale1_down_convbottleneck_BatchNorm_beta =      MSDNet_Layer3_4[1].modules[0].modules[1].modules[1].bias
Msdnet_b2_step3_scale1_down_convbottleneck_BatchNorm_gamma =     MSDNet_Layer3_4[1].modules[0].modules[1].modules[1].weight
Msdnet_b2_step3_scale1_down_convdowm_weights =                   MSDNet_Layer3_4[1].modules[0].modules[1].modules[3].weight
Msdnet_b2_step3_scale1_down_convdowm_BatchNorm_beta =            MSDNet_Layer3_4[1].modules[0].modules[1].modules[4].bias
Msdnet_b2_step3_scale1_down_convdowm_BatchNorm_gamma =           MSDNet_Layer3_4[1].modules[0].modules[1].modules[4].weight
Msdnet_b2_step3_scale1_normal_convbottleneck_weights =           MSDNet_Layer3_4[1].modules[0].modules[2].modules[0].weight
Msdnet_b2_step3_scale1_normal_convbottleneck_BatchNorm_beta =    MSDNet_Layer3_4[1].modules[0].modules[2].modules[1].bias
Msdnet_b2_step3_scale1_normal_convbottleneck_BatchNorm_gamma =   MSDNet_Layer3_4[1].modules[0].modules[2].modules[1].weight
Msdnet_b2_step3_scale1_normal_convnormal_weights =               MSDNet_Layer3_4[1].modules[0].modules[2].modules[3].weight
Msdnet_b2_step3_scale1_normal_convnormal_BatchNorm_beta =        MSDNet_Layer3_4[1].modules[0].modules[2].modules[4].bias
Msdnet_b2_step3_scale1_normal_convnormal_BatchNorm_gamma =       MSDNet_Layer3_4[1].modules[0].modules[2].modules[4].weight
#block4
block4 = Joint[3].modules
MSDNet_Layer4_1 = block4[0].modules
Msdnet_b3_step0_scale0_normal_convbottleneck_weights =           MSDNet_Layer4_1[0].modules[0].modules[1].modules[0].weight
Msdnet_b3_step0_scale0_normal_convbottleneck_BatchNorm_beta =    MSDNet_Layer4_1[0].modules[0].modules[1].modules[1].bias
Msdnet_b3_step0_scale0_normal_convbottleneck_BatchNorm_gamma =   MSDNet_Layer4_1[0].modules[0].modules[1].modules[1].weight
Msdnet_b3_step0_scale0_normal_convnormal_weights =               MSDNet_Layer4_1[0].modules[0].modules[1].modules[3].weight
Msdnet_b3_step0_scale0_normal_convnormal_BatchNorm_beta =        MSDNet_Layer4_1[0].modules[0].modules[1].modules[4].bias
Msdnet_b3_step0_scale0_normal_convnormal_BatchNorm_gamma =       MSDNet_Layer4_1[0].modules[0].modules[1].modules[4].weight
Msdnet_b3_step0_scale1_down_convbottleneck_weights =             MSDNet_Layer4_1[1].modules[0].modules[1].modules[0].weight
Msdnet_b3_step0_scale1_down_convbottleneck_BatchNorm_beta =      MSDNet_Layer4_1[1].modules[0].modules[1].modules[1].bias
Msdnet_b3_step0_scale1_down_convbottleneck_BatchNorm_gamma =     MSDNet_Layer4_1[1].modules[0].modules[1].modules[1].weight
Msdnet_b3_step0_scale1_down_convdowm_weights =                   MSDNet_Layer4_1[1].modules[0].modules[1].modules[3].weight
Msdnet_b3_step0_scale1_down_convdowm_BatchNorm_beta =            MSDNet_Layer4_1[1].modules[0].modules[1].modules[4].bias
Msdnet_b3_step0_scale1_down_convdowm_BatchNorm_gamma =           MSDNet_Layer4_1[1].modules[0].modules[1].modules[4].weight
Msdnet_b3_step0_scale1_normal_convbottleneck_weights =           MSDNet_Layer4_1[1].modules[0].modules[2].modules[0].weight
Msdnet_b3_step0_scale1_normal_convbottleneck_BatchNorm_beta =    MSDNet_Layer4_1[1].modules[0].modules[2].modules[1].bias
Msdnet_b3_step0_scale1_normal_convbottleneck_BatchNorm_gamma =   MSDNet_Layer4_1[1].modules[0].modules[2].modules[1].weight
Msdnet_b3_step0_scale1_normal_convnormal_weights =               MSDNet_Layer4_1[1].modules[0].modules[2].modules[3].weight
Msdnet_b3_step0_scale1_normal_convnormal_BatchNorm_beta =        MSDNet_Layer4_1[1].modules[0].modules[2].modules[4].bias
Msdnet_b3_step0_scale1_normal_convnormal_BatchNorm_gamma =       MSDNet_Layer4_1[1].modules[0].modules[2].modules[4].weight

MSDNet_Layer4_2 = block4[1].modules
Msdnet_b3_step1_scale0_normal_convbottleneck_weights =           MSDNet_Layer4_2[0].modules[0].modules[1].modules[0].weight
Msdnet_b3_step1_scale0_normal_convbottleneck_BatchNorm_beta =    MSDNet_Layer4_2[0].modules[0].modules[1].modules[1].bias
Msdnet_b3_step1_scale0_normal_convbottleneck_BatchNorm_gamma =   MSDNet_Layer4_2[0].modules[0].modules[1].modules[1].weight
Msdnet_b3_step1_scale0_normal_convnormal_weights =               MSDNet_Layer4_2[0].modules[0].modules[1].modules[3].weight
Msdnet_b3_step1_scale0_normal_convnormal_BatchNorm_beta =        MSDNet_Layer4_2[0].modules[0].modules[1].modules[4].bias
Msdnet_b3_step1_scale0_normal_convnormal_BatchNorm_gamma =       MSDNet_Layer4_2[0].modules[0].modules[1].modules[4].weight
Msdnet_b3_step1_scale1_down_convbottleneck_weights =             MSDNet_Layer4_2[1].modules[0].modules[1].modules[0].weight
Msdnet_b3_step1_scale1_down_convbottleneck_BatchNorm_beta =      MSDNet_Layer4_2[1].modules[0].modules[1].modules[1].bias
Msdnet_b3_step1_scale1_down_convbottleneck_BatchNorm_gamma =     MSDNet_Layer4_2[1].modules[0].modules[1].modules[1].weight
Msdnet_b3_step1_scale1_down_convdowm_weights =                   MSDNet_Layer4_2[1].modules[0].modules[1].modules[3].weight
Msdnet_b3_step1_scale1_down_convdowm_BatchNorm_beta =            MSDNet_Layer4_2[1].modules[0].modules[1].modules[4].bias
Msdnet_b3_step1_scale1_down_convdowm_BatchNorm_gamma =           MSDNet_Layer4_2[1].modules[0].modules[1].modules[4].weight
Msdnet_b3_step1_scale1_normal_convbottleneck_weights =           MSDNet_Layer4_2[1].modules[0].modules[2].modules[0].weight
Msdnet_b3_step1_scale1_normal_convbottleneck_BatchNorm_beta =    MSDNet_Layer4_2[1].modules[0].modules[2].modules[1].bias
Msdnet_b3_step1_scale1_normal_convbottleneck_BatchNorm_gamma =   MSDNet_Layer4_2[1].modules[0].modules[2].modules[1].weight
Msdnet_b3_step1_scale1_normal_convnormal_weights =               MSDNet_Layer4_2[1].modules[0].modules[2].modules[3].weight
Msdnet_b3_step1_scale1_normal_convnormal_BatchNorm_beta =        MSDNet_Layer4_2[1].modules[0].modules[2].modules[4].bias
Msdnet_b3_step1_scale1_normal_convnormal_BatchNorm_gamma =       MSDNet_Layer4_2[1].modules[0].modules[2].modules[4].weight

MSDNet_Layer4_3 = block4[2].modules
Msdnet_b3_step2_scale0_normal_convbottleneck_weights =           MSDNet_Layer4_3[0].modules[0].modules[1].modules[0].weight
Msdnet_b3_step2_scale0_normal_convbottleneck_BatchNorm_beta =    MSDNet_Layer4_3[0].modules[0].modules[1].modules[1].bias
Msdnet_b3_step2_scale0_normal_convbottleneck_BatchNorm_gamma =   MSDNet_Layer4_3[0].modules[0].modules[1].modules[1].weight
Msdnet_b3_step2_scale0_normal_convnormal_weights =               MSDNet_Layer4_3[0].modules[0].modules[1].modules[3].weight
Msdnet_b3_step2_scale0_normal_convnormal_BatchNorm_beta =        MSDNet_Layer4_3[0].modules[0].modules[1].modules[4].bias
Msdnet_b3_step2_scale0_normal_convnormal_BatchNorm_gamma =       MSDNet_Layer4_3[0].modules[0].modules[1].modules[4].weight
Msdnet_b3_step2_scale1_down_convbottleneck_weights =             MSDNet_Layer4_3[1].modules[0].modules[1].modules[0].weight
Msdnet_b3_step2_scale1_down_convbottleneck_BatchNorm_beta =      MSDNet_Layer4_3[1].modules[0].modules[1].modules[1].bias
Msdnet_b3_step2_scale1_down_convbottleneck_BatchNorm_gamma =     MSDNet_Layer4_3[1].modules[0].modules[1].modules[1].weight
Msdnet_b3_step2_scale1_down_convdowm_weights =                   MSDNet_Layer4_3[1].modules[0].modules[1].modules[3].weight
Msdnet_b3_step2_scale1_down_convdowm_BatchNorm_beta =            MSDNet_Layer4_3[1].modules[0].modules[1].modules[4].bias
Msdnet_b3_step2_scale1_down_convdowm_BatchNorm_gamma =           MSDNet_Layer4_3[1].modules[0].modules[1].modules[4].weight
Msdnet_b3_step2_scale1_normal_convbottleneck_weights =           MSDNet_Layer4_3[1].modules[0].modules[2].modules[0].weight
Msdnet_b3_step2_scale1_normal_convbottleneck_BatchNorm_beta =    MSDNet_Layer4_3[1].modules[0].modules[2].modules[1].bias
Msdnet_b3_step2_scale1_normal_convbottleneck_BatchNorm_gamma =   MSDNet_Layer4_3[1].modules[0].modules[2].modules[1].weight
Msdnet_b3_step2_scale1_normal_convnormal_weights =               MSDNet_Layer4_3[1].modules[0].modules[2].modules[3].weight
Msdnet_b3_step2_scale1_normal_convnormal_BatchNorm_beta =        MSDNet_Layer4_3[1].modules[0].modules[2].modules[4].bias
Msdnet_b3_step2_scale1_normal_convnormal_BatchNorm_gamma =       MSDNet_Layer4_3[1].modules[0].modules[2].modules[4].weight

MSDNet_Layer4_4 = block4[3].modules
Msdnet_b3_step3_scale0_down_convbottleneck_weights =             MSDNet_Layer4_4[0].modules[0].modules[1].modules[0].weight
Msdnet_b3_step3_scale0_down_convbottleneck_BatchNorm_beta =      MSDNet_Layer4_4[0].modules[0].modules[1].modules[1].bias
Msdnet_b3_step3_scale0_down_convbottleneck_BatchNorm_gamma =     MSDNet_Layer4_4[0].modules[0].modules[1].modules[1].weight
Msdnet_b3_step3_scale0_down_convdowm_weights =                   MSDNet_Layer4_4[0].modules[0].modules[1].modules[3].weight
Msdnet_b3_step3_scale0_down_convdowm_BatchNorm_beta =            MSDNet_Layer4_4[0].modules[0].modules[1].modules[4].bias
Msdnet_b3_step3_scale0_down_convdowm_BatchNorm_gamma =           MSDNet_Layer4_4[0].modules[0].modules[1].modules[4].weight
Msdnet_b3_step3_scale0_normal_convbottleneck_weights =           MSDNet_Layer4_4[0].modules[0].modules[2].modules[0].weight
Msdnet_b3_step3_scale0_normal_convbottleneck_BatchNorm_beta =    MSDNet_Layer4_4[0].modules[0].modules[2].modules[1].bias
Msdnet_b3_step3_scale0_normal_convbottleneck_BatchNorm_gamma =   MSDNet_Layer4_4[0].modules[0].modules[2].modules[1].weight
Msdnet_b3_step3_scale0_normal_convnormal_weights =               MSDNet_Layer4_4[0].modules[0].modules[2].modules[3].weight
Msdnet_b3_step3_scale0_normal_convnormal_BatchNorm_beta =        MSDNet_Layer4_4[0].modules[0].modules[2].modules[4].bias
Msdnet_b3_step3_scale0_normal_convnormal_BatchNorm_gamma =       MSDNet_Layer4_4[0].modules[0].modules[2].modules[4].weight

Trans4 = block4[4].modules
Msdnet_b3_transition_scale0_conv1_weights =      Trans4[0].modules[0].weight
Msdnet_b3_transition_scale0_conv1_BatchNorm_beta =Trans4[0].modules[1].bias
Msdnet_b3_transition_scale0_conv1_BatchNorm_gamma =Trans4[0].modules[1].weight

#block5
block5 = Joint[4].modules
MSDNet_Layer5_1 = block5[0].modules
Msdnet_b4_step0_scale0_normal_convbottleneck_weights =           MSDNet_Layer5_1[0].modules[0].modules[1].modules[0].weight
Msdnet_b4_step0_scale0_normal_convbottleneck_BatchNorm_beta =    MSDNet_Layer5_1[0].modules[0].modules[1].modules[1].bias
Msdnet_b4_step0_scale0_normal_convbottleneck_BatchNorm_gamma =   MSDNet_Layer5_1[0].modules[0].modules[1].modules[1].weight
Msdnet_b4_step0_scale0_normal_convnormal_weights =               MSDNet_Layer5_1[0].modules[0].modules[1].modules[3].weight
Msdnet_b4_step0_scale0_normal_convnormal_BatchNorm_beta =        MSDNet_Layer5_1[0].modules[0].modules[1].modules[4].bias
Msdnet_b4_step0_scale0_normal_convnormal_BatchNorm_gamma =       MSDNet_Layer5_1[0].modules[0].modules[1].modules[4].weight

MSDNet_Layer5_2 = block5[1].modules
Msdnet_b4_step1_scale0_normal_convbottleneck_weights =           MSDNet_Layer5_2[0].modules[0].modules[1].modules[0].weight
Msdnet_b4_step1_scale0_normal_convbottleneck_BatchNorm_beta =    MSDNet_Layer5_2[0].modules[0].modules[1].modules[1].bias
Msdnet_b4_step1_scale0_normal_convbottleneck_BatchNorm_gamma =   MSDNet_Layer5_2[0].modules[0].modules[1].modules[1].weight
Msdnet_b4_step1_scale0_normal_convnormal_weights =               MSDNet_Layer5_2[0].modules[0].modules[1].modules[3].weight
Msdnet_b4_step1_scale0_normal_convnormal_BatchNorm_beta =        MSDNet_Layer5_2[0].modules[0].modules[1].modules[4].bias
Msdnet_b4_step1_scale0_normal_convnormal_BatchNorm_gamma =       MSDNet_Layer5_2[0].modules[0].modules[1].modules[4].weight

MSDNet_Layer5_3 = block5[2].modules
Msdnet_b4_step2_scale0_normal_convbottleneck_weights =           MSDNet_Layer5_3[0].modules[0].modules[1].modules[0].weight
Msdnet_b4_step2_scale0_normal_convbottleneck_BatchNorm_beta =    MSDNet_Layer5_3[0].modules[0].modules[1].modules[1].bias
Msdnet_b4_step2_scale0_normal_convbottleneck_BatchNorm_gamma =   MSDNet_Layer5_3[0].modules[0].modules[1].modules[1].weight
Msdnet_b4_step2_scale0_normal_convnormal_weights =               MSDNet_Layer5_3[0].modules[0].modules[1].modules[3].weight
Msdnet_b4_step2_scale0_normal_convnormal_BatchNorm_beta =        MSDNet_Layer5_3[0].modules[0].modules[1].modules[4].bias
Msdnet_b4_step2_scale0_normal_convnormal_BatchNorm_gamma =       MSDNet_Layer5_3[0].modules[0].modules[1].modules[4].weight

MSDNet_Layer5_4 = block5[3].modules
Msdnet_b4_step3_scale0_normal_convbottleneck_weights =           MSDNet_Layer5_4[0].modules[0].modules[1].modules[0].weight
Msdnet_b4_step3_scale0_normal_convbottleneck_BatchNorm_beta =    MSDNet_Layer5_4[0].modules[0].modules[1].modules[1].bias
Msdnet_b4_step3_scale0_normal_convbottleneck_BatchNorm_gamma =   MSDNet_Layer5_4[0].modules[0].modules[1].modules[1].weight
Msdnet_b4_step3_scale0_normal_convnormal_weights =               MSDNet_Layer5_4[0].modules[0].modules[1].modules[3].weight
Msdnet_b4_step3_scale0_normal_convnormal_BatchNorm_beta =        MSDNet_Layer5_4[0].modules[0].modules[1].modules[4].bias
Msdnet_b4_step3_scale0_normal_convnormal_BatchNorm_gamma =       MSDNet_Layer5_4[0].modules[0].modules[1].modules[4].weight

#convbox_detection_Conv_weights =
#convbox_detection_Conv_biases =


model_weights_temp={
'Msdnet/b0/scale0/conv1/weights':Msdnet_b0_scale0_conv1_weights,
'Msdnet/b0/scale0/conv1/BatchNorm/beta':Msdnet_b0_scale0_conv1_BatchNorm_beta,
'Msdnet/b0/scale0/conv1/BatchNorm/gamma':Msdnet_b0_scale0_conv1_BatchNorm_gamma,
'Msdnet/b0/scale1/conv1/weights':Msdnet_b0_scale1_conv1_weights,
'Msdnet/b0/scale1/conv1/BatchNorm/beta':Msdnet_b0_scale1_conv1_BatchNorm_beta,
'Msdnet/b0/scale1/conv1/BatchNorm/gamma':Msdnet_b0_scale1_conv1_BatchNorm_gamma,
'Msdnet/b0/scale2/conv1/weights':Msdnet_b0_scale2_conv1_weights,
'Msdnet/b0/scale2/conv1/BatchNorm/beta':Msdnet_b0_scale2_conv1_BatchNorm_beta,
'Msdnet/b0/scale2/conv1/BatchNorm/gamma':Msdnet_b0_scale2_conv1_BatchNorm_gamma,
'Msdnet/b0/scale3/conv1/weights':Msdnet_b0_scale3_conv1_weights,
'Msdnet/b0/scale3/conv1/BatchNorm/beta':Msdnet_b0_scale3_conv1_BatchNorm_beta,
'Msdnet/b0/scale3/conv1/BatchNorm/gamma':Msdnet_b0_scale3_conv1_BatchNorm_gamma,
'Msdnet/b0/step0/scale0/normal/convbottleneck/weights':Msdnet_b0_step0_scale0_normal_convbottleneck_weights,
'Msdnet/b0/step0/scale0/normal/convbottleneck/BatchNorm/beta':Msdnet_b0_step0_scale0_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b0/step0/scale0/normal/convbottleneck/BatchNorm/gamma':Msdnet_b0_step0_scale0_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b0/step0/scale0/normal/convnormal/weights':Msdnet_b0_step0_scale0_normal_convnormal_weights,
'Msdnet/b0/step0/scale0/normal/convnormal/BatchNorm/beta':Msdnet_b0_step0_scale0_normal_convnormal_BatchNorm_beta,
'Msdnet/b0/step0/scale0/normal/convnormal/BatchNorm/gamma':Msdnet_b0_step0_scale0_normal_convnormal_BatchNorm_gamma,
'Msdnet/b0/step0/scale1/down/convbottleneck/weights':Msdnet_b0_step0_scale1_down_convbottleneck_weights,
'Msdnet/b0/step0/scale1/down/convbottleneck/BatchNorm/beta':Msdnet_b0_step0_scale1_down_convbottleneck_BatchNorm_beta,
'Msdnet/b0/step0/scale1/down/convbottleneck/BatchNorm/gamma':Msdnet_b0_step0_scale1_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b0/step0/scale1/down/convdowm/weights':Msdnet_b0_step0_scale1_down_convdowm_weights,
'Msdnet/b0/step0/scale1/down/convdowm/BatchNorm/beta':Msdnet_b0_step0_scale1_down_convdowm_BatchNorm_beta,
'Msdnet/b0/step0/scale1/down/convdowm/BatchNorm/gamma':Msdnet_b0_step0_scale1_down_convdowm_BatchNorm_gamma,
'Msdnet/b0/step0/scale1/normal/convbottleneck/weights':Msdnet_b0_step0_scale1_normal_convbottleneck_weights,
'Msdnet/b0/step0/scale1/normal/convbottleneck/BatchNorm/beta':Msdnet_b0_step0_scale1_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b0/step0/scale1/normal/convbottleneck/BatchNorm/gamma':Msdnet_b0_step0_scale1_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b0/step0/scale1/normal/convnormal/weights':Msdnet_b0_step0_scale1_normal_convnormal_weights,
'Msdnet/b0/step0/scale1/normal/convnormal/BatchNorm/beta':Msdnet_b0_step0_scale1_normal_convnormal_BatchNorm_beta,
'Msdnet/b0/step0/scale1/normal/convnormal/BatchNorm/gamma':Msdnet_b0_step0_scale1_normal_convnormal_BatchNorm_gamma,
'Msdnet/b0/step0/scale2/down/convbottleneck/weights':Msdnet_b0_step0_scale2_down_convbottleneck_weights,
'Msdnet/b0/step0/scale2/down/convbottleneck/BatchNorm/beta':Msdnet_b0_step0_scale2_down_convbottleneck_BatchNorm_beta,
'Msdnet/b0/step0/scale2/down/convbottleneck/BatchNorm/gamma':Msdnet_b0_step0_scale2_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b0/step0/scale2/down/convdowm/weights':Msdnet_b0_step0_scale2_down_convdowm_weights,
'Msdnet/b0/step0/scale2/down/convdowm/BatchNorm/beta':Msdnet_b0_step0_scale2_down_convdowm_BatchNorm_beta,
'Msdnet/b0/step0/scale2/down/convdowm/BatchNorm/gamma':Msdnet_b0_step0_scale2_down_convdowm_BatchNorm_gamma,
'Msdnet/b0/step0/scale2/normal/convbottleneck/weights':Msdnet_b0_step0_scale2_normal_convbottleneck_weights,
'Msdnet/b0/step0/scale2/normal/convbottleneck/BatchNorm/beta':Msdnet_b0_step0_scale2_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b0/step0/scale2/normal/convbottleneck/BatchNorm/gamma':Msdnet_b0_step0_scale2_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b0/step0/scale2/normal/convnormal/weights':Msdnet_b0_step0_scale2_normal_convnormal_weights,
'Msdnet/b0/step0/scale2/normal/convnormal/BatchNorm/beta':Msdnet_b0_step0_scale2_normal_convnormal_BatchNorm_beta,
'Msdnet/b0/step0/scale2/normal/convnormal/BatchNorm/gamma':Msdnet_b0_step0_scale2_normal_convnormal_BatchNorm_gamma,
'Msdnet/b0/step0/scale3/down/convbottleneck/weights':Msdnet_b0_step0_scale3_down_convbottleneck_weights,
'Msdnet/b0/step0/scale3/down/convbottleneck/BatchNorm/beta':Msdnet_b0_step0_scale3_down_convbottleneck_BatchNorm_beta,
'Msdnet/b0/step0/scale3/down/convbottleneck/BatchNorm/gamma':Msdnet_b0_step0_scale3_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b0/step0/scale3/down/convdowm/weights':Msdnet_b0_step0_scale3_down_convdowm_weights,
'Msdnet/b0/step0/scale3/down/convdowm/BatchNorm/beta':Msdnet_b0_step0_scale3_down_convdowm_BatchNorm_beta,
'Msdnet/b0/step0/scale3/down/convdowm/BatchNorm/gamma':Msdnet_b0_step0_scale3_down_convdowm_BatchNorm_gamma,
'Msdnet/b0/step0/scale3/normal/convbottleneck/weights':Msdnet_b0_step0_scale3_normal_convbottleneck_weights,
'Msdnet/b0/step0/scale3/normal/convbottleneck/BatchNorm/beta':Msdnet_b0_step0_scale3_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b0/step0/scale3/normal/convbottleneck/BatchNorm/gamma':Msdnet_b0_step0_scale3_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b0/step0/scale3/normal/convnormal/weights':Msdnet_b0_step0_scale3_normal_convnormal_weights,
'Msdnet/b0/step0/scale3/normal/convnormal/BatchNorm/beta':Msdnet_b0_step0_scale3_normal_convnormal_BatchNorm_beta,
'Msdnet/b0/step0/scale3/normal/convnormal/BatchNorm/gamma':Msdnet_b0_step0_scale3_normal_convnormal_BatchNorm_gamma,
'Msdnet/b0/step1/scale0/normal/convbottleneck/weights':Msdnet_b0_step1_scale0_normal_convbottleneck_weights,
'Msdnet/b0/step1/scale0/normal/convbottleneck/BatchNorm/beta':Msdnet_b0_step1_scale0_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b0/step1/scale0/normal/convbottleneck/BatchNorm/gamma':Msdnet_b0_step1_scale0_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b0/step1/scale0/normal/convnormal/weights':Msdnet_b0_step1_scale0_normal_convnormal_weights,
'Msdnet/b0/step1/scale0/normal/convnormal/BatchNorm/beta':Msdnet_b0_step1_scale0_normal_convnormal_BatchNorm_beta,
'Msdnet/b0/step1/scale0/normal/convnormal/BatchNorm/gamma':Msdnet_b0_step1_scale0_normal_convnormal_BatchNorm_gamma,
'Msdnet/b0/step1/scale1/down/convbottleneck/weights':Msdnet_b0_step1_scale1_down_convbottleneck_weights,
'Msdnet/b0/step1/scale1/down/convbottleneck/BatchNorm/beta':Msdnet_b0_step1_scale1_down_convbottleneck_BatchNorm_beta,
'Msdnet/b0/step1/scale1/down/convbottleneck/BatchNorm/gamma':Msdnet_b0_step1_scale1_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b0/step1/scale1/down/convdowm/weights':Msdnet_b0_step1_scale1_down_convdowm_weights,
'Msdnet/b0/step1/scale1/down/convdowm/BatchNorm/beta':Msdnet_b0_step1_scale1_down_convdowm_BatchNorm_beta,
'Msdnet/b0/step1/scale1/down/convdowm/BatchNorm/gamma':Msdnet_b0_step1_scale1_down_convdowm_BatchNorm_gamma,
'Msdnet/b0/step1/scale1/normal/convbottleneck/weights':Msdnet_b0_step1_scale1_normal_convbottleneck_weights,
'Msdnet/b0/step1/scale1/normal/convbottleneck/BatchNorm/beta':Msdnet_b0_step1_scale1_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b0/step1/scale1/normal/convbottleneck/BatchNorm/gamma':Msdnet_b0_step1_scale1_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b0/step1/scale1/normal/convnormal/weights':Msdnet_b0_step1_scale1_normal_convnormal_weights,
'Msdnet/b0/step1/scale1/normal/convnormal/BatchNorm/beta':Msdnet_b0_step1_scale1_normal_convnormal_BatchNorm_beta,
'Msdnet/b0/step1/scale1/normal/convnormal/BatchNorm/gamma':Msdnet_b0_step1_scale1_normal_convnormal_BatchNorm_gamma,
'Msdnet/b0/step1/scale2/down/convbottleneck/weights':Msdnet_b0_step1_scale2_down_convbottleneck_weights,
'Msdnet/b0/step1/scale2/down/convbottleneck/BatchNorm/beta':Msdnet_b0_step1_scale2_down_convbottleneck_BatchNorm_beta,
'Msdnet/b0/step1/scale2/down/convbottleneck/BatchNorm/gamma':Msdnet_b0_step1_scale2_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b0/step1/scale2/down/convdowm/weights':Msdnet_b0_step1_scale2_down_convdowm_weights,
'Msdnet/b0/step1/scale2/down/convdowm/BatchNorm/beta':Msdnet_b0_step1_scale2_down_convdowm_BatchNorm_beta,
'Msdnet/b0/step1/scale2/down/convdowm/BatchNorm/gamma':Msdnet_b0_step1_scale2_down_convdowm_BatchNorm_gamma,
'Msdnet/b0/step1/scale2/normal/convbottleneck/weights':Msdnet_b0_step1_scale2_normal_convbottleneck_weights,
'Msdnet/b0/step1/scale2/normal/convbottleneck/BatchNorm/beta':Msdnet_b0_step1_scale2_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b0/step1/scale2/normal/convbottleneck/BatchNorm/gamma':Msdnet_b0_step1_scale2_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b0/step1/scale2/normal/convnormal/weights':Msdnet_b0_step1_scale2_normal_convnormal_weights,
'Msdnet/b0/step1/scale2/normal/convnormal/BatchNorm/beta':Msdnet_b0_step1_scale2_normal_convnormal_BatchNorm_beta,
'Msdnet/b0/step1/scale2/normal/convnormal/BatchNorm/gamma':Msdnet_b0_step1_scale2_normal_convnormal_BatchNorm_gamma,
'Msdnet/b0/step1/scale3/down/convbottleneck/weights':Msdnet_b0_step1_scale3_down_convbottleneck_weights,
'Msdnet/b0/step1/scale3/down/convbottleneck/BatchNorm/beta':Msdnet_b0_step1_scale3_down_convbottleneck_BatchNorm_beta,
'Msdnet/b0/step1/scale3/down/convbottleneck/BatchNorm/gamma':Msdnet_b0_step1_scale3_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b0/step1/scale3/down/convdowm/weights':Msdnet_b0_step1_scale3_down_convdowm_weights,
'Msdnet/b0/step1/scale3/down/convdowm/BatchNorm/beta':Msdnet_b0_step1_scale3_down_convdowm_BatchNorm_beta,
'Msdnet/b0/step1/scale3/down/convdowm/BatchNorm/gamma':Msdnet_b0_step1_scale3_down_convdowm_BatchNorm_gamma,
'Msdnet/b0/step1/scale3/normal/convbottleneck/weights':Msdnet_b0_step1_scale3_normal_convbottleneck_weights,
'Msdnet/b0/step1/scale3/normal/convbottleneck/BatchNorm/beta':Msdnet_b0_step1_scale3_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b0/step1/scale3/normal/convbottleneck/BatchNorm/gamma':Msdnet_b0_step1_scale3_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b0/step1/scale3/normal/convnormal/weights':Msdnet_b0_step1_scale3_normal_convnormal_weights,
'Msdnet/b0/step1/scale3/normal/convnormal/BatchNorm/beta':Msdnet_b0_step1_scale3_normal_convnormal_BatchNorm_beta,
'Msdnet/b0/step1/scale3/normal/convnormal/BatchNorm/gamma':Msdnet_b0_step1_scale3_normal_convnormal_BatchNorm_gamma,
'Msdnet/b0/step2/scale0/normal/convbottleneck/weights':Msdnet_b0_step2_scale0_normal_convbottleneck_weights,
'Msdnet/b0/step2/scale0/normal/convbottleneck/BatchNorm/beta':Msdnet_b0_step2_scale0_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b0/step2/scale0/normal/convbottleneck/BatchNorm/gamma':Msdnet_b0_step2_scale0_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b0/step2/scale0/normal/convnormal/weights':Msdnet_b0_step2_scale0_normal_convnormal_weights,
'Msdnet/b0/step2/scale0/normal/convnormal/BatchNorm/beta':Msdnet_b0_step2_scale0_normal_convnormal_BatchNorm_beta,
'Msdnet/b0/step2/scale0/normal/convnormal/BatchNorm/gamma':Msdnet_b0_step2_scale0_normal_convnormal_BatchNorm_gamma,
'Msdnet/b0/step2/scale1/down/convbottleneck/weights':Msdnet_b0_step2_scale1_down_convbottleneck_weights,
'Msdnet/b0/step2/scale1/down/convbottleneck/BatchNorm/beta':Msdnet_b0_step2_scale1_down_convbottleneck_BatchNorm_beta,
'Msdnet/b0/step2/scale1/down/convbottleneck/BatchNorm/gamma':Msdnet_b0_step2_scale1_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b0/step2/scale1/down/convdowm/weights':Msdnet_b0_step2_scale1_down_convdowm_weights,
'Msdnet/b0/step2/scale1/down/convdowm/BatchNorm/beta':Msdnet_b0_step2_scale1_down_convdowm_BatchNorm_beta,
'Msdnet/b0/step2/scale1/down/convdowm/BatchNorm/gamma':Msdnet_b0_step2_scale1_down_convdowm_BatchNorm_gamma,
'Msdnet/b0/step2/scale1/normal/convbottleneck/weights':Msdnet_b0_step2_scale1_normal_convbottleneck_weights,
'Msdnet/b0/step2/scale1/normal/convbottleneck/BatchNorm/beta':Msdnet_b0_step2_scale1_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b0/step2/scale1/normal/convbottleneck/BatchNorm/gamma':Msdnet_b0_step2_scale1_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b0/step2/scale1/normal/convnormal/weights':Msdnet_b0_step2_scale1_normal_convnormal_weights,
'Msdnet/b0/step2/scale1/normal/convnormal/BatchNorm/beta':Msdnet_b0_step2_scale1_normal_convnormal_BatchNorm_beta,
'Msdnet/b0/step2/scale1/normal/convnormal/BatchNorm/gamma':Msdnet_b0_step2_scale1_normal_convnormal_BatchNorm_gamma,
'Msdnet/b0/step2/scale2/down/convbottleneck/weights':Msdnet_b0_step2_scale2_down_convbottleneck_weights,
'Msdnet/b0/step2/scale2/down/convbottleneck/BatchNorm/beta':Msdnet_b0_step2_scale2_down_convbottleneck_BatchNorm_beta,
'Msdnet/b0/step2/scale2/down/convbottleneck/BatchNorm/gamma':Msdnet_b0_step2_scale2_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b0/step2/scale2/down/convdowm/weights':Msdnet_b0_step2_scale2_down_convdowm_weights,
'Msdnet/b0/step2/scale2/down/convdowm/BatchNorm/beta':Msdnet_b0_step2_scale2_down_convdowm_BatchNorm_beta,
'Msdnet/b0/step2/scale2/down/convdowm/BatchNorm/gamma':Msdnet_b0_step2_scale2_down_convdowm_BatchNorm_gamma,
'Msdnet/b0/step2/scale2/normal/convbottleneck/weights':Msdnet_b0_step2_scale2_normal_convbottleneck_weights,
'Msdnet/b0/step2/scale2/normal/convbottleneck/BatchNorm/beta':Msdnet_b0_step2_scale2_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b0/step2/scale2/normal/convbottleneck/BatchNorm/gamma':Msdnet_b0_step2_scale2_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b0/step2/scale2/normal/convnormal/weights':Msdnet_b0_step2_scale2_normal_convnormal_weights,
'Msdnet/b0/step2/scale2/normal/convnormal/BatchNorm/beta':Msdnet_b0_step2_scale2_normal_convnormal_BatchNorm_beta,
'Msdnet/b0/step2/scale2/normal/convnormal/BatchNorm/gamma':Msdnet_b0_step2_scale2_normal_convnormal_BatchNorm_gamma,
'Msdnet/b0/step2/scale3/down/convbottleneck/weights':Msdnet_b0_step2_scale3_down_convbottleneck_weights,
'Msdnet/b0/step2/scale3/down/convbottleneck/BatchNorm/beta':Msdnet_b0_step2_scale3_down_convbottleneck_BatchNorm_beta,
'Msdnet/b0/step2/scale3/down/convbottleneck/BatchNorm/gamma':Msdnet_b0_step2_scale3_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b0/step2/scale3/down/convdowm/weights':Msdnet_b0_step2_scale3_down_convdowm_weights,
'Msdnet/b0/step2/scale3/down/convdowm/BatchNorm/beta':Msdnet_b0_step2_scale3_down_convdowm_BatchNorm_beta,
'Msdnet/b0/step2/scale3/down/convdowm/BatchNorm/gamma':Msdnet_b0_step2_scale3_down_convdowm_BatchNorm_gamma,
'Msdnet/b0/step2/scale3/normal/convbottleneck/weights':Msdnet_b0_step2_scale3_normal_convbottleneck_weights,
'Msdnet/b0/step2/scale3/normal/convbottleneck/BatchNorm/beta':Msdnet_b0_step2_scale3_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b0/step2/scale3/normal/convbottleneck/BatchNorm/gamma':Msdnet_b0_step2_scale3_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b0/step2/scale3/normal/convnormal/weights':Msdnet_b0_step2_scale3_normal_convnormal_weights,
'Msdnet/b0/step2/scale3/normal/convnormal/BatchNorm/beta':Msdnet_b0_step2_scale3_normal_convnormal_BatchNorm_beta,
'Msdnet/b0/step2/scale3/normal/convnormal/BatchNorm/gamma':Msdnet_b0_step2_scale3_normal_convnormal_BatchNorm_gamma,
'Msdnet/b0/step3/scale0/normal/convbottleneck/weights':Msdnet_b0_step3_scale0_normal_convbottleneck_weights,
'Msdnet/b0/step3/scale0/normal/convbottleneck/BatchNorm/beta':Msdnet_b0_step3_scale0_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b0/step3/scale0/normal/convbottleneck/BatchNorm/gamma':Msdnet_b0_step3_scale0_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b0/step3/scale0/normal/convnormal/weights':Msdnet_b0_step3_scale0_normal_convnormal_weights,
'Msdnet/b0/step3/scale0/normal/convnormal/BatchNorm/beta':Msdnet_b0_step3_scale0_normal_convnormal_BatchNorm_beta,
'Msdnet/b0/step3/scale0/normal/convnormal/BatchNorm/gamma':Msdnet_b0_step3_scale0_normal_convnormal_BatchNorm_gamma,
'Msdnet/b0/step3/scale1/down/convbottleneck/weights':Msdnet_b0_step3_scale1_down_convbottleneck_weights,
'Msdnet/b0/step3/scale1/down/convbottleneck/BatchNorm/beta':Msdnet_b0_step3_scale1_down_convbottleneck_BatchNorm_beta,
'Msdnet/b0/step3/scale1/down/convbottleneck/BatchNorm/gamma':Msdnet_b0_step3_scale1_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b0/step3/scale1/down/convdowm/weights':Msdnet_b0_step3_scale1_down_convdowm_weights,
'Msdnet/b0/step3/scale1/down/convdowm/BatchNorm/beta':Msdnet_b0_step3_scale1_down_convdowm_BatchNorm_beta,
'Msdnet/b0/step3/scale1/down/convdowm/BatchNorm/gamma':Msdnet_b0_step3_scale1_down_convdowm_BatchNorm_gamma,
'Msdnet/b0/step3/scale1/normal/convbottleneck/weights':Msdnet_b0_step3_scale1_normal_convbottleneck_weights,
'Msdnet/b0/step3/scale1/normal/convbottleneck/BatchNorm/beta':Msdnet_b0_step3_scale1_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b0/step3/scale1/normal/convbottleneck/BatchNorm/gamma':Msdnet_b0_step3_scale1_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b0/step3/scale1/normal/convnormal/weights':Msdnet_b0_step3_scale1_normal_convnormal_weights,
'Msdnet/b0/step3/scale1/normal/convnormal/BatchNorm/beta':Msdnet_b0_step3_scale1_normal_convnormal_BatchNorm_beta,
'Msdnet/b0/step3/scale1/normal/convnormal/BatchNorm/gamma':Msdnet_b0_step3_scale1_normal_convnormal_BatchNorm_gamma,
'Msdnet/b0/step3/scale2/down/convbottleneck/weights':Msdnet_b0_step3_scale2_down_convbottleneck_weights,
'Msdnet/b0/step3/scale2/down/convbottleneck/BatchNorm/beta':Msdnet_b0_step3_scale2_down_convbottleneck_BatchNorm_beta,
'Msdnet/b0/step3/scale2/down/convbottleneck/BatchNorm/gamma':Msdnet_b0_step3_scale2_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b0/step3/scale2/down/convdowm/weights':Msdnet_b0_step3_scale2_down_convdowm_weights,
'Msdnet/b0/step3/scale2/down/convdowm/BatchNorm/beta':Msdnet_b0_step3_scale2_down_convdowm_BatchNorm_beta,
'Msdnet/b0/step3/scale2/down/convdowm/BatchNorm/gamma':Msdnet_b0_step3_scale2_down_convdowm_BatchNorm_gamma,
'Msdnet/b0/step3/scale2/normal/convbottleneck/weights':Msdnet_b0_step3_scale2_normal_convbottleneck_weights,
'Msdnet/b0/step3/scale2/normal/convbottleneck/BatchNorm/beta':Msdnet_b0_step3_scale2_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b0/step3/scale2/normal/convbottleneck/BatchNorm/gamma':Msdnet_b0_step3_scale2_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b0/step3/scale2/normal/convnormal/weights':Msdnet_b0_step3_scale2_normal_convnormal_weights,
'Msdnet/b0/step3/scale2/normal/convnormal/BatchNorm/beta':Msdnet_b0_step3_scale2_normal_convnormal_BatchNorm_beta,
'Msdnet/b0/step3/scale2/normal/convnormal/BatchNorm/gamma':Msdnet_b0_step3_scale2_normal_convnormal_BatchNorm_gamma,
'Msdnet/b0/step3/scale3/down/convbottleneck/weights':Msdnet_b0_step3_scale3_down_convbottleneck_weights,
'Msdnet/b0/step3/scale3/down/convbottleneck/BatchNorm/beta':Msdnet_b0_step3_scale3_down_convbottleneck_BatchNorm_beta,
'Msdnet/b0/step3/scale3/down/convbottleneck/BatchNorm/gamma':Msdnet_b0_step3_scale3_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b0/step3/scale3/down/convdowm/weights':Msdnet_b0_step3_scale3_down_convdowm_weights,
'Msdnet/b0/step3/scale3/down/convdowm/BatchNorm/beta':Msdnet_b0_step3_scale3_down_convdowm_BatchNorm_beta,
'Msdnet/b0/step3/scale3/down/convdowm/BatchNorm/gamma':Msdnet_b0_step3_scale3_down_convdowm_BatchNorm_gamma,
'Msdnet/b0/step3/scale3/normal/convbottleneck/weights':Msdnet_b0_step3_scale3_normal_convbottleneck_weights,
'Msdnet/b0/step3/scale3/normal/convbottleneck/BatchNorm/beta':Msdnet_b0_step3_scale3_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b0/step3/scale3/normal/convbottleneck/BatchNorm/gamma':Msdnet_b0_step3_scale3_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b0/step3/scale3/normal/convnormal/weights':Msdnet_b0_step3_scale3_normal_convnormal_weights,
'Msdnet/b0/step3/scale3/normal/convnormal/BatchNorm/beta':Msdnet_b0_step3_scale3_normal_convnormal_BatchNorm_beta,
'Msdnet/b0/step3/scale3/normal/convnormal/BatchNorm/gamma':Msdnet_b0_step3_scale3_normal_convnormal_BatchNorm_gamma,
'Msdnet/b1/step0/scale0/normal/convbottleneck/weights':Msdnet_b1_step0_scale0_normal_convbottleneck_weights,
'Msdnet/b1/step0/scale0/normal/convbottleneck/BatchNorm/beta':Msdnet_b1_step0_scale0_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b1/step0/scale0/normal/convbottleneck/BatchNorm/gamma':Msdnet_b1_step0_scale0_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b1/step0/scale0/normal/convnormal/weights':Msdnet_b1_step0_scale0_normal_convnormal_weights,
'Msdnet/b1/step0/scale0/normal/convnormal/BatchNorm/beta':Msdnet_b1_step0_scale0_normal_convnormal_BatchNorm_beta,
'Msdnet/b1/step0/scale0/normal/convnormal/BatchNorm/gamma':Msdnet_b1_step0_scale0_normal_convnormal_BatchNorm_gamma,
'Msdnet/b1/step0/scale1/down/convbottleneck/weights':Msdnet_b1_step0_scale1_down_convbottleneck_weights,
'Msdnet/b1/step0/scale1/down/convbottleneck/BatchNorm/beta':Msdnet_b1_step0_scale1_down_convbottleneck_BatchNorm_beta,
'Msdnet/b1/step0/scale1/down/convbottleneck/BatchNorm/gamma':Msdnet_b1_step0_scale1_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b1/step0/scale1/down/convdowm/weights':Msdnet_b1_step0_scale1_down_convdowm_weights,
'Msdnet/b1/step0/scale1/down/convdowm/BatchNorm/beta':Msdnet_b1_step0_scale1_down_convdowm_BatchNorm_beta,
'Msdnet/b1/step0/scale1/down/convdowm/BatchNorm/gamma':Msdnet_b1_step0_scale1_down_convdowm_BatchNorm_gamma,
'Msdnet/b1/step0/scale1/normal/convbottleneck/weights':Msdnet_b1_step0_scale1_normal_convbottleneck_weights,
'Msdnet/b1/step0/scale1/normal/convbottleneck/BatchNorm/beta':Msdnet_b1_step0_scale1_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b1/step0/scale1/normal/convbottleneck/BatchNorm/gamma':Msdnet_b1_step0_scale1_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b1/step0/scale1/normal/convnormal/weights':Msdnet_b1_step0_scale1_normal_convnormal_weights,
'Msdnet/b1/step0/scale1/normal/convnormal/BatchNorm/beta':Msdnet_b1_step0_scale1_normal_convnormal_BatchNorm_beta,
'Msdnet/b1/step0/scale1/normal/convnormal/BatchNorm/gamma':Msdnet_b1_step0_scale1_normal_convnormal_BatchNorm_gamma,
'Msdnet/b1/step0/scale2/down/convbottleneck/weights':Msdnet_b1_step0_scale2_down_convbottleneck_weights,
'Msdnet/b1/step0/scale2/down/convbottleneck/BatchNorm/beta':Msdnet_b1_step0_scale2_down_convbottleneck_BatchNorm_beta,
'Msdnet/b1/step0/scale2/down/convbottleneck/BatchNorm/gamma':Msdnet_b1_step0_scale2_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b1/step0/scale2/down/convdowm/weights':Msdnet_b1_step0_scale2_down_convdowm_weights,
'Msdnet/b1/step0/scale2/down/convdowm/BatchNorm/beta':Msdnet_b1_step0_scale2_down_convdowm_BatchNorm_beta,
'Msdnet/b1/step0/scale2/down/convdowm/BatchNorm/gamma':Msdnet_b1_step0_scale2_down_convdowm_BatchNorm_gamma,
'Msdnet/b1/step0/scale2/normal/convbottleneck/weights':Msdnet_b1_step0_scale2_normal_convbottleneck_weights,
'Msdnet/b1/step0/scale2/normal/convbottleneck/BatchNorm/beta':Msdnet_b1_step0_scale2_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b1/step0/scale2/normal/convbottleneck/BatchNorm/gamma':Msdnet_b1_step0_scale2_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b1/step0/scale2/normal/convnormal/weights':Msdnet_b1_step0_scale2_normal_convnormal_weights,
'Msdnet/b1/step0/scale2/normal/convnormal/BatchNorm/beta':Msdnet_b1_step0_scale2_normal_convnormal_BatchNorm_beta,
'Msdnet/b1/step0/scale2/normal/convnormal/BatchNorm/gamma':Msdnet_b1_step0_scale2_normal_convnormal_BatchNorm_gamma,
'Msdnet/b1/step0/scale3/down/convbottleneck/weights':Msdnet_b1_step0_scale3_down_convbottleneck_weights,
'Msdnet/b1/step0/scale3/down/convbottleneck/BatchNorm/beta':Msdnet_b1_step0_scale3_down_convbottleneck_BatchNorm_beta,
'Msdnet/b1/step0/scale3/down/convbottleneck/BatchNorm/gamma':Msdnet_b1_step0_scale3_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b1/step0/scale3/down/convdowm/weights':Msdnet_b1_step0_scale3_down_convdowm_weights,
'Msdnet/b1/step0/scale3/down/convdowm/BatchNorm/beta':Msdnet_b1_step0_scale3_down_convdowm_BatchNorm_beta,
'Msdnet/b1/step0/scale3/down/convdowm/BatchNorm/gamma':Msdnet_b1_step0_scale3_down_convdowm_BatchNorm_gamma,
'Msdnet/b1/step0/scale3/normal/convbottleneck/weights':Msdnet_b1_step0_scale3_normal_convbottleneck_weights,
'Msdnet/b1/step0/scale3/normal/convbottleneck/BatchNorm/beta':Msdnet_b1_step0_scale3_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b1/step0/scale3/normal/convbottleneck/BatchNorm/gamma':Msdnet_b1_step0_scale3_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b1/step0/scale3/normal/convnormal/weights':Msdnet_b1_step0_scale3_normal_convnormal_weights,
'Msdnet/b1/step0/scale3/normal/convnormal/BatchNorm/beta':Msdnet_b1_step0_scale3_normal_convnormal_BatchNorm_beta,
'Msdnet/b1/step0/scale3/normal/convnormal/BatchNorm/gamma':Msdnet_b1_step0_scale3_normal_convnormal_BatchNorm_gamma,
'Msdnet/b1/step1/scale0/down/convbottleneck/weights':Msdnet_b1_step1_scale0_down_convbottleneck_weights,
'Msdnet/b1/step1/scale0/down/convbottleneck/BatchNorm/beta':Msdnet_b1_step1_scale0_down_convbottleneck_BatchNorm_beta,
'Msdnet/b1/step1/scale0/down/convbottleneck/BatchNorm/gamma':Msdnet_b1_step1_scale0_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b1/step1/scale0/down/convdowm/weights':Msdnet_b1_step1_scale0_down_convdowm_weights,
'Msdnet/b1/step1/scale0/down/convdowm/BatchNorm/beta':Msdnet_b1_step1_scale0_down_convdowm_BatchNorm_beta,
'Msdnet/b1/step1/scale0/down/convdowm/BatchNorm/gamma':Msdnet_b1_step1_scale0_down_convdowm_BatchNorm_gamma,
'Msdnet/b1/step1/scale0/normal/convbottleneck/weights':Msdnet_b1_step1_scale0_normal_convbottleneck_weights,
'Msdnet/b1/step1/scale0/normal/convbottleneck/BatchNorm/beta':Msdnet_b1_step1_scale0_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b1/step1/scale0/normal/convbottleneck/BatchNorm/gamma':Msdnet_b1_step1_scale0_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b1/step1/scale0/normal/convnormal/weights':Msdnet_b1_step1_scale0_normal_convnormal_weights,
'Msdnet/b1/step1/scale0/normal/convnormal/BatchNorm/beta':Msdnet_b1_step1_scale0_normal_convnormal_BatchNorm_beta,
'Msdnet/b1/step1/scale0/normal/convnormal/BatchNorm/gamma':Msdnet_b1_step1_scale0_normal_convnormal_BatchNorm_gamma,
'Msdnet/b1/step1/scale1/down/convbottleneck/weights':Msdnet_b1_step1_scale1_down_convbottleneck_weights,
'Msdnet/b1/step1/scale1/down/convbottleneck/BatchNorm/beta':Msdnet_b1_step1_scale1_down_convbottleneck_BatchNorm_beta,
'Msdnet/b1/step1/scale1/down/convbottleneck/BatchNorm/gamma':Msdnet_b1_step1_scale1_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b1/step1/scale1/down/convdowm/weights':Msdnet_b1_step1_scale1_down_convdowm_weights,
'Msdnet/b1/step1/scale1/down/convdowm/BatchNorm/beta':Msdnet_b1_step1_scale1_down_convdowm_BatchNorm_beta,
'Msdnet/b1/step1/scale1/down/convdowm/BatchNorm/gamma':Msdnet_b1_step1_scale1_down_convdowm_BatchNorm_gamma,
'Msdnet/b1/step1/scale1/normal/convbottleneck/weights':Msdnet_b1_step1_scale1_normal_convbottleneck_weights,
'Msdnet/b1/step1/scale1/normal/convbottleneck/BatchNorm/beta':Msdnet_b1_step1_scale1_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b1/step1/scale1/normal/convbottleneck/BatchNorm/gamma':Msdnet_b1_step1_scale1_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b1/step1/scale1/normal/convnormal/weights':Msdnet_b1_step1_scale1_normal_convnormal_weights,
'Msdnet/b1/step1/scale1/normal/convnormal/BatchNorm/beta':Msdnet_b1_step1_scale1_normal_convnormal_BatchNorm_beta,
'Msdnet/b1/step1/scale1/normal/convnormal/BatchNorm/gamma':Msdnet_b1_step1_scale1_normal_convnormal_BatchNorm_gamma,
'Msdnet/b1/step1/scale2/down/convbottleneck/weights':Msdnet_b1_step1_scale2_down_convbottleneck_weights,
'Msdnet/b1/step1/scale2/down/convbottleneck/BatchNorm/beta':Msdnet_b1_step1_scale2_down_convbottleneck_BatchNorm_beta,
'Msdnet/b1/step1/scale2/down/convbottleneck/BatchNorm/gamma':Msdnet_b1_step1_scale2_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b1/step1/scale2/down/convdowm/weights':Msdnet_b1_step1_scale2_down_convdowm_weights,
'Msdnet/b1/step1/scale2/down/convdowm/BatchNorm/beta':Msdnet_b1_step1_scale2_down_convdowm_BatchNorm_beta,
'Msdnet/b1/step1/scale2/down/convdowm/BatchNorm/gamma':Msdnet_b1_step1_scale2_down_convdowm_BatchNorm_gamma,
'Msdnet/b1/step1/scale2/normal/convbottleneck/weights':Msdnet_b1_step1_scale2_normal_convbottleneck_weights,
'Msdnet/b1/step1/scale2/normal/convbottleneck/BatchNorm/beta':Msdnet_b1_step1_scale2_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b1/step1/scale2/normal/convbottleneck/BatchNorm/gamma':Msdnet_b1_step1_scale2_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b1/step1/scale2/normal/convnormal/weights':Msdnet_b1_step1_scale2_normal_convnormal_weights,
'Msdnet/b1/step1/scale2/normal/convnormal/BatchNorm/beta':Msdnet_b1_step1_scale2_normal_convnormal_BatchNorm_beta,
'Msdnet/b1/step1/scale2/normal/convnormal/BatchNorm/gamma':Msdnet_b1_step1_scale2_normal_convnormal_BatchNorm_gamma,
'Msdnet/b1/transition/scale0/conv1/weights':Msdnet_b1_transition_scale0_conv1_weights,
'Msdnet/b1/transition/scale0/conv1/BatchNorm/beta':Msdnet_b1_transition_scale0_conv1_BatchNorm_beta,
'Msdnet/b1/transition/scale0/conv1/BatchNorm/gamma':Msdnet_b1_transition_scale0_conv1_BatchNorm_gamma,
'Msdnet/b1/transition/scale1/conv1/weights':Msdnet_b1_transition_scale1_conv1_weights,
'Msdnet/b1/transition/scale1/conv1/BatchNorm/beta':Msdnet_b1_transition_scale1_conv1_BatchNorm_beta,
'Msdnet/b1/transition/scale1/conv1/BatchNorm/gamma':Msdnet_b1_transition_scale1_conv1_BatchNorm_gamma,
'Msdnet/b1/transition/scale2/conv1/weights':Msdnet_b1_transition_scale2_conv1_weights,
'Msdnet/b1/transition/scale2/conv1/BatchNorm/beta':Msdnet_b1_transition_scale2_conv1_BatchNorm_beta,
'Msdnet/b1/transition/scale2/conv1/BatchNorm/gamma':Msdnet_b1_transition_scale2_conv1_BatchNorm_gamma,
'Msdnet/b1/step2/scale0/normal/convbottleneck/weights':Msdnet_b1_step2_scale0_normal_convbottleneck_weights,
'Msdnet/b1/step2/scale0/normal/convbottleneck/BatchNorm/beta':Msdnet_b1_step2_scale0_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b1/step2/scale0/normal/convbottleneck/BatchNorm/gamma':Msdnet_b1_step2_scale0_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b1/step2/scale0/normal/convnormal/weights':Msdnet_b1_step2_scale0_normal_convnormal_weights,
'Msdnet/b1/step2/scale0/normal/convnormal/BatchNorm/beta':Msdnet_b1_step2_scale0_normal_convnormal_BatchNorm_beta,
'Msdnet/b1/step2/scale0/normal/convnormal/BatchNorm/gamma':Msdnet_b1_step2_scale0_normal_convnormal_BatchNorm_gamma,
'Msdnet/b1/step2/scale1/down/convbottleneck/weights':Msdnet_b1_step2_scale1_down_convbottleneck_weights,
'Msdnet/b1/step2/scale1/down/convbottleneck/BatchNorm/beta':Msdnet_b1_step2_scale1_down_convbottleneck_BatchNorm_beta,
'Msdnet/b1/step2/scale1/down/convbottleneck/BatchNorm/gamma':Msdnet_b1_step2_scale1_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b1/step2/scale1/down/convdowm/weights':Msdnet_b1_step2_scale1_down_convdowm_weights,
'Msdnet/b1/step2/scale1/down/convdowm/BatchNorm/beta':Msdnet_b1_step2_scale1_down_convdowm_BatchNorm_beta,
'Msdnet/b1/step2/scale1/down/convdowm/BatchNorm/gamma':Msdnet_b1_step2_scale1_down_convdowm_BatchNorm_gamma,
'Msdnet/b1/step2/scale1/normal/convbottleneck/weights':Msdnet_b1_step2_scale1_normal_convbottleneck_weights,
'Msdnet/b1/step2/scale1/normal/convbottleneck/BatchNorm/beta':Msdnet_b1_step2_scale1_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b1/step2/scale1/normal/convbottleneck/BatchNorm/gamma':Msdnet_b1_step2_scale1_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b1/step2/scale1/normal/convnormal/weights':Msdnet_b1_step2_scale1_normal_convnormal_weights,
'Msdnet/b1/step2/scale1/normal/convnormal/BatchNorm/beta':Msdnet_b1_step2_scale1_normal_convnormal_BatchNorm_beta,
'Msdnet/b1/step2/scale1/normal/convnormal/BatchNorm/gamma':Msdnet_b1_step2_scale1_normal_convnormal_BatchNorm_gamma,
'Msdnet/b1/step2/scale2/down/convbottleneck/weights':Msdnet_b1_step2_scale2_down_convbottleneck_weights,
'Msdnet/b1/step2/scale2/down/convbottleneck/BatchNorm/beta':Msdnet_b1_step2_scale2_down_convbottleneck_BatchNorm_beta,
'Msdnet/b1/step2/scale2/down/convbottleneck/BatchNorm/gamma':Msdnet_b1_step2_scale2_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b1/step2/scale2/down/convdowm/weights':Msdnet_b1_step2_scale2_down_convdowm_weights,
'Msdnet/b1/step2/scale2/down/convdowm/BatchNorm/beta':Msdnet_b1_step2_scale2_down_convdowm_BatchNorm_beta,
'Msdnet/b1/step2/scale2/down/convdowm/BatchNorm/gamma':Msdnet_b1_step2_scale2_down_convdowm_BatchNorm_gamma,
'Msdnet/b1/step2/scale2/normal/convbottleneck/weights':Msdnet_b1_step2_scale2_normal_convbottleneck_weights,
'Msdnet/b1/step2/scale2/normal/convbottleneck/BatchNorm/beta':Msdnet_b1_step2_scale2_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b1/step2/scale2/normal/convbottleneck/BatchNorm/gamma':Msdnet_b1_step2_scale2_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b1/step2/scale2/normal/convnormal/weights':Msdnet_b1_step2_scale2_normal_convnormal_weights,
'Msdnet/b1/step2/scale2/normal/convnormal/BatchNorm/beta':Msdnet_b1_step2_scale2_normal_convnormal_BatchNorm_beta,
'Msdnet/b1/step2/scale2/normal/convnormal/BatchNorm/gamma':Msdnet_b1_step2_scale2_normal_convnormal_BatchNorm_gamma,
'Msdnet/b1/step3/scale0/normal/convbottleneck/weights':Msdnet_b1_step3_scale0_normal_convbottleneck_weights,
'Msdnet/b1/step3/scale0/normal/convbottleneck/BatchNorm/beta':Msdnet_b1_step3_scale0_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b1/step3/scale0/normal/convbottleneck/BatchNorm/gamma':Msdnet_b1_step3_scale0_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b1/step3/scale0/normal/convnormal/weights':Msdnet_b1_step3_scale0_normal_convnormal_weights,
'Msdnet/b1/step3/scale0/normal/convnormal/BatchNorm/beta':Msdnet_b1_step3_scale0_normal_convnormal_BatchNorm_beta,
'Msdnet/b1/step3/scale0/normal/convnormal/BatchNorm/gamma':Msdnet_b1_step3_scale0_normal_convnormal_BatchNorm_gamma,
'Msdnet/b1/step3/scale1/down/convbottleneck/weights':Msdnet_b1_step3_scale1_down_convbottleneck_weights,
'Msdnet/b1/step3/scale1/down/convbottleneck/BatchNorm/beta':Msdnet_b1_step3_scale1_down_convbottleneck_BatchNorm_beta,
'Msdnet/b1/step3/scale1/down/convbottleneck/BatchNorm/gamma':Msdnet_b1_step3_scale1_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b1/step3/scale1/down/convdowm/weights':Msdnet_b1_step3_scale1_down_convdowm_weights,
'Msdnet/b1/step3/scale1/down/convdowm/BatchNorm/beta':Msdnet_b1_step3_scale1_down_convdowm_BatchNorm_beta,
'Msdnet/b1/step3/scale1/down/convdowm/BatchNorm/gamma':Msdnet_b1_step3_scale1_down_convdowm_BatchNorm_gamma,
'Msdnet/b1/step3/scale1/normal/convbottleneck/weights':Msdnet_b1_step3_scale1_normal_convbottleneck_weights,
'Msdnet/b1/step3/scale1/normal/convbottleneck/BatchNorm/beta':Msdnet_b1_step3_scale1_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b1/step3/scale1/normal/convbottleneck/BatchNorm/gamma':Msdnet_b1_step3_scale1_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b1/step3/scale1/normal/convnormal/weights':Msdnet_b1_step3_scale1_normal_convnormal_weights,
'Msdnet/b1/step3/scale1/normal/convnormal/BatchNorm/beta':Msdnet_b1_step3_scale1_normal_convnormal_BatchNorm_beta,
'Msdnet/b1/step3/scale1/normal/convnormal/BatchNorm/gamma':Msdnet_b1_step3_scale1_normal_convnormal_BatchNorm_gamma,
'Msdnet/b1/step3/scale2/down/convbottleneck/weights':Msdnet_b1_step3_scale2_down_convbottleneck_weights,
'Msdnet/b1/step3/scale2/down/convbottleneck/BatchNorm/beta':Msdnet_b1_step3_scale2_down_convbottleneck_BatchNorm_beta,
'Msdnet/b1/step3/scale2/down/convbottleneck/BatchNorm/gamma':Msdnet_b1_step3_scale2_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b1/step3/scale2/down/convdowm/weights':Msdnet_b1_step3_scale2_down_convdowm_weights,
'Msdnet/b1/step3/scale2/down/convdowm/BatchNorm/beta':Msdnet_b1_step3_scale2_down_convdowm_BatchNorm_beta,
'Msdnet/b1/step3/scale2/down/convdowm/BatchNorm/gamma':Msdnet_b1_step3_scale2_down_convdowm_BatchNorm_gamma,
'Msdnet/b1/step3/scale2/normal/convbottleneck/weights':Msdnet_b1_step3_scale2_normal_convbottleneck_weights,
'Msdnet/b1/step3/scale2/normal/convbottleneck/BatchNorm/beta':Msdnet_b1_step3_scale2_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b1/step3/scale2/normal/convbottleneck/BatchNorm/gamma':Msdnet_b1_step3_scale2_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b1/step3/scale2/normal/convnormal/weights':Msdnet_b1_step3_scale2_normal_convnormal_weights,
'Msdnet/b1/step3/scale2/normal/convnormal/BatchNorm/beta':Msdnet_b1_step3_scale2_normal_convnormal_BatchNorm_beta,
'Msdnet/b1/step3/scale2/normal/convnormal/BatchNorm/gamma':Msdnet_b1_step3_scale2_normal_convnormal_BatchNorm_gamma,
'Msdnet/b2/step0/scale0/normal/convbottleneck/weights':Msdnet_b2_step0_scale0_normal_convbottleneck_weights,
'Msdnet/b2/step0/scale0/normal/convbottleneck/BatchNorm/beta':Msdnet_b2_step0_scale0_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b2/step0/scale0/normal/convbottleneck/BatchNorm/gamma':Msdnet_b2_step0_scale0_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b2/step0/scale0/normal/convnormal/weights':Msdnet_b2_step0_scale0_normal_convnormal_weights,
'Msdnet/b2/step0/scale0/normal/convnormal/BatchNorm/beta':Msdnet_b2_step0_scale0_normal_convnormal_BatchNorm_beta,
'Msdnet/b2/step0/scale0/normal/convnormal/BatchNorm/gamma':Msdnet_b2_step0_scale0_normal_convnormal_BatchNorm_gamma,
'Msdnet/b2/step0/scale1/down/convbottleneck/weights':Msdnet_b2_step0_scale1_down_convbottleneck_weights,
'Msdnet/b2/step0/scale1/down/convbottleneck/BatchNorm/beta':Msdnet_b2_step0_scale1_down_convbottleneck_BatchNorm_beta,
'Msdnet/b2/step0/scale1/down/convbottleneck/BatchNorm/gamma':Msdnet_b2_step0_scale1_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b2/step0/scale1/down/convdowm/weights':Msdnet_b2_step0_scale1_down_convdowm_weights,
'Msdnet/b2/step0/scale1/down/convdowm/BatchNorm/beta':Msdnet_b2_step0_scale1_down_convdowm_BatchNorm_beta,
'Msdnet/b2/step0/scale1/down/convdowm/BatchNorm/gamma':Msdnet_b2_step0_scale1_down_convdowm_BatchNorm_gamma,
'Msdnet/b2/step0/scale1/normal/convbottleneck/weights':Msdnet_b2_step0_scale1_normal_convbottleneck_weights,
'Msdnet/b2/step0/scale1/normal/convbottleneck/BatchNorm/beta':Msdnet_b2_step0_scale1_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b2/step0/scale1/normal/convbottleneck/BatchNorm/gamma':Msdnet_b2_step0_scale1_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b2/step0/scale1/normal/convnormal/weights':Msdnet_b2_step0_scale1_normal_convnormal_weights,
'Msdnet/b2/step0/scale1/normal/convnormal/BatchNorm/beta':Msdnet_b2_step0_scale1_normal_convnormal_BatchNorm_beta,
'Msdnet/b2/step0/scale1/normal/convnormal/BatchNorm/gamma':Msdnet_b2_step0_scale1_normal_convnormal_BatchNorm_gamma,
'Msdnet/b2/step0/scale2/down/convbottleneck/weights':Msdnet_b2_step0_scale2_down_convbottleneck_weights,
'Msdnet/b2/step0/scale2/down/convbottleneck/BatchNorm/beta':Msdnet_b2_step0_scale2_down_convbottleneck_BatchNorm_beta,
'Msdnet/b2/step0/scale2/down/convbottleneck/BatchNorm/gamma':Msdnet_b2_step0_scale2_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b2/step0/scale2/down/convdowm/weights':Msdnet_b2_step0_scale2_down_convdowm_weights,
'Msdnet/b2/step0/scale2/down/convdowm/BatchNorm/beta':Msdnet_b2_step0_scale2_down_convdowm_BatchNorm_beta,
'Msdnet/b2/step0/scale2/down/convdowm/BatchNorm/gamma':Msdnet_b2_step0_scale2_down_convdowm_BatchNorm_gamma,
'Msdnet/b2/step0/scale2/normal/convbottleneck/weights':Msdnet_b2_step0_scale2_normal_convbottleneck_weights,
'Msdnet/b2/step0/scale2/normal/convbottleneck/BatchNorm/beta':Msdnet_b2_step0_scale2_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b2/step0/scale2/normal/convbottleneck/BatchNorm/gamma':Msdnet_b2_step0_scale2_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b2/step0/scale2/normal/convnormal/weights':Msdnet_b2_step0_scale2_normal_convnormal_weights,
'Msdnet/b2/step0/scale2/normal/convnormal/BatchNorm/beta':Msdnet_b2_step0_scale2_normal_convnormal_BatchNorm_beta,
'Msdnet/b2/step0/scale2/normal/convnormal/BatchNorm/gamma':Msdnet_b2_step0_scale2_normal_convnormal_BatchNorm_gamma,
'Msdnet/b2/step1/scale0/normal/convbottleneck/weights':Msdnet_b2_step1_scale0_normal_convbottleneck_weights,
'Msdnet/b2/step1/scale0/normal/convbottleneck/BatchNorm/beta':Msdnet_b2_step1_scale0_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b2/step1/scale0/normal/convbottleneck/BatchNorm/gamma':Msdnet_b2_step1_scale0_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b2/step1/scale0/normal/convnormal/weights':Msdnet_b2_step1_scale0_normal_convnormal_weights,
'Msdnet/b2/step1/scale0/normal/convnormal/BatchNorm/beta':Msdnet_b2_step1_scale0_normal_convnormal_BatchNorm_beta,
'Msdnet/b2/step1/scale0/normal/convnormal/BatchNorm/gamma':Msdnet_b2_step1_scale0_normal_convnormal_BatchNorm_gamma,
'Msdnet/b2/step1/scale1/down/convbottleneck/weights':Msdnet_b2_step1_scale1_down_convbottleneck_weights,
'Msdnet/b2/step1/scale1/down/convbottleneck/BatchNorm/beta':Msdnet_b2_step1_scale1_down_convbottleneck_BatchNorm_beta,
'Msdnet/b2/step1/scale1/down/convbottleneck/BatchNorm/gamma':Msdnet_b2_step1_scale1_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b2/step1/scale1/down/convdowm/weights':Msdnet_b2_step1_scale1_down_convdowm_weights,
'Msdnet/b2/step1/scale1/down/convdowm/BatchNorm/beta':Msdnet_b2_step1_scale1_down_convdowm_BatchNorm_beta,
'Msdnet/b2/step1/scale1/down/convdowm/BatchNorm/gamma':Msdnet_b2_step1_scale1_down_convdowm_BatchNorm_gamma,
'Msdnet/b2/step1/scale1/normal/convbottleneck/weights':Msdnet_b2_step1_scale1_normal_convbottleneck_weights,
'Msdnet/b2/step1/scale1/normal/convbottleneck/BatchNorm/beta':Msdnet_b2_step1_scale1_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b2/step1/scale1/normal/convbottleneck/BatchNorm/gamma':Msdnet_b2_step1_scale1_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b2/step1/scale1/normal/convnormal/weights':Msdnet_b2_step1_scale1_normal_convnormal_weights,
'Msdnet/b2/step1/scale1/normal/convnormal/BatchNorm/beta':Msdnet_b2_step1_scale1_normal_convnormal_BatchNorm_beta,
'Msdnet/b2/step1/scale1/normal/convnormal/BatchNorm/gamma':Msdnet_b2_step1_scale1_normal_convnormal_BatchNorm_gamma,
'Msdnet/b2/step1/scale2/down/convbottleneck/weights':Msdnet_b2_step1_scale2_down_convbottleneck_weights,
'Msdnet/b2/step1/scale2/down/convbottleneck/BatchNorm/beta':Msdnet_b2_step1_scale2_down_convbottleneck_BatchNorm_beta,
'Msdnet/b2/step1/scale2/down/convbottleneck/BatchNorm/gamma':Msdnet_b2_step1_scale2_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b2/step1/scale2/down/convdowm/weights':Msdnet_b2_step1_scale2_down_convdowm_weights,
'Msdnet/b2/step1/scale2/down/convdowm/BatchNorm/beta':Msdnet_b2_step1_scale2_down_convdowm_BatchNorm_beta,
'Msdnet/b2/step1/scale2/down/convdowm/BatchNorm/gamma':Msdnet_b2_step1_scale2_down_convdowm_BatchNorm_gamma,
'Msdnet/b2/step1/scale2/normal/convbottleneck/weights':Msdnet_b2_step1_scale2_normal_convbottleneck_weights,
'Msdnet/b2/step1/scale2/normal/convbottleneck/BatchNorm/beta':Msdnet_b2_step1_scale2_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b2/step1/scale2/normal/convbottleneck/BatchNorm/gamma':Msdnet_b2_step1_scale2_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b2/step1/scale2/normal/convnormal/weights':Msdnet_b2_step1_scale2_normal_convnormal_weights,
'Msdnet/b2/step1/scale2/normal/convnormal/BatchNorm/beta':Msdnet_b2_step1_scale2_normal_convnormal_BatchNorm_beta,
'Msdnet/b2/step1/scale2/normal/convnormal/BatchNorm/gamma':Msdnet_b2_step1_scale2_normal_convnormal_BatchNorm_gamma,
'Msdnet/b2/step2/scale0/down/convbottleneck/weights':Msdnet_b2_step2_scale0_down_convbottleneck_weights,
'Msdnet/b2/step2/scale0/down/convbottleneck/BatchNorm/beta':Msdnet_b2_step2_scale0_down_convbottleneck_BatchNorm_beta,
'Msdnet/b2/step2/scale0/down/convbottleneck/BatchNorm/gamma':Msdnet_b2_step2_scale0_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b2/step2/scale0/down/convdowm/weights':Msdnet_b2_step2_scale0_down_convdowm_weights,
'Msdnet/b2/step2/scale0/down/convdowm/BatchNorm/beta':Msdnet_b2_step2_scale0_down_convdowm_BatchNorm_beta,
'Msdnet/b2/step2/scale0/down/convdowm/BatchNorm/gamma':Msdnet_b2_step2_scale0_down_convdowm_BatchNorm_gamma,
'Msdnet/b2/step2/scale0/normal/convbottleneck/weights':Msdnet_b2_step2_scale0_normal_convbottleneck_weights,
'Msdnet/b2/step2/scale0/normal/convbottleneck/BatchNorm/beta':Msdnet_b2_step2_scale0_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b2/step2/scale0/normal/convbottleneck/BatchNorm/gamma':Msdnet_b2_step2_scale0_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b2/step2/scale0/normal/convnormal/weights':Msdnet_b2_step2_scale0_normal_convnormal_weights,
'Msdnet/b2/step2/scale0/normal/convnormal/BatchNorm/beta':Msdnet_b2_step2_scale0_normal_convnormal_BatchNorm_beta,
'Msdnet/b2/step2/scale0/normal/convnormal/BatchNorm/gamma':Msdnet_b2_step2_scale0_normal_convnormal_BatchNorm_gamma,
'Msdnet/b2/step2/scale1/down/convbottleneck/weights':Msdnet_b2_step2_scale1_down_convbottleneck_weights,
'Msdnet/b2/step2/scale1/down/convbottleneck/BatchNorm/beta':Msdnet_b2_step2_scale1_down_convbottleneck_BatchNorm_beta,
'Msdnet/b2/step2/scale1/down/convbottleneck/BatchNorm/gamma':Msdnet_b2_step2_scale1_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b2/step2/scale1/down/convdowm/weights':Msdnet_b2_step2_scale1_down_convdowm_weights,
'Msdnet/b2/step2/scale1/down/convdowm/BatchNorm/beta':Msdnet_b2_step2_scale1_down_convdowm_BatchNorm_beta,
'Msdnet/b2/step2/scale1/down/convdowm/BatchNorm/gamma':Msdnet_b2_step2_scale1_down_convdowm_BatchNorm_gamma,
'Msdnet/b2/step2/scale1/normal/convbottleneck/weights':Msdnet_b2_step2_scale1_normal_convbottleneck_weights,
'Msdnet/b2/step2/scale1/normal/convbottleneck/BatchNorm/beta':Msdnet_b2_step2_scale1_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b2/step2/scale1/normal/convbottleneck/BatchNorm/gamma':Msdnet_b2_step2_scale1_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b2/step2/scale1/normal/convnormal/weights':Msdnet_b2_step2_scale1_normal_convnormal_weights,
'Msdnet/b2/step2/scale1/normal/convnormal/BatchNorm/beta':Msdnet_b2_step2_scale1_normal_convnormal_BatchNorm_beta,
'Msdnet/b2/step2/scale1/normal/convnormal/BatchNorm/gamma':Msdnet_b2_step2_scale1_normal_convnormal_BatchNorm_gamma,
'Msdnet/b2/transition/scale0/conv1/weights':Msdnet_b2_transition_scale0_conv1_weights,
'Msdnet/b2/transition/scale0/conv1/BatchNorm/beta':Msdnet_b2_transition_scale0_conv1_BatchNorm_beta,
'Msdnet/b2/transition/scale0/conv1/BatchNorm/gamma':Msdnet_b2_transition_scale0_conv1_BatchNorm_gamma,
'Msdnet/b2/transition/scale1/conv1/weights':Msdnet_b2_transition_scale1_conv1_weights,
'Msdnet/b2/transition/scale1/conv1/BatchNorm/beta':Msdnet_b2_transition_scale1_conv1_BatchNorm_beta,
'Msdnet/b2/transition/scale1/conv1/BatchNorm/gamma':Msdnet_b2_transition_scale1_conv1_BatchNorm_gamma,
'Msdnet/b2/step3/scale0/normal/convbottleneck/weights':Msdnet_b2_step3_scale0_normal_convbottleneck_weights,
'Msdnet/b2/step3/scale0/normal/convbottleneck/BatchNorm/beta':Msdnet_b2_step3_scale0_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b2/step3/scale0/normal/convbottleneck/BatchNorm/gamma':Msdnet_b2_step3_scale0_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b2/step3/scale0/normal/convnormal/weights':Msdnet_b2_step3_scale0_normal_convnormal_weights,
'Msdnet/b2/step3/scale0/normal/convnormal/BatchNorm/beta':Msdnet_b2_step3_scale0_normal_convnormal_BatchNorm_beta,
'Msdnet/b2/step3/scale0/normal/convnormal/BatchNorm/gamma':Msdnet_b2_step3_scale0_normal_convnormal_BatchNorm_gamma,
'Msdnet/b2/step3/scale1/down/convbottleneck/weights':Msdnet_b2_step3_scale1_down_convbottleneck_weights,
'Msdnet/b2/step3/scale1/down/convbottleneck/BatchNorm/beta':Msdnet_b2_step3_scale1_down_convbottleneck_BatchNorm_beta,
'Msdnet/b2/step3/scale1/down/convbottleneck/BatchNorm/gamma':Msdnet_b2_step3_scale1_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b2/step3/scale1/down/convdowm/weights':Msdnet_b2_step3_scale1_down_convdowm_weights,
'Msdnet/b2/step3/scale1/down/convdowm/BatchNorm/beta':Msdnet_b2_step3_scale1_down_convdowm_BatchNorm_beta,
'Msdnet/b2/step3/scale1/down/convdowm/BatchNorm/gamma':Msdnet_b2_step3_scale1_down_convdowm_BatchNorm_gamma,
'Msdnet/b2/step3/scale1/normal/convbottleneck/weights':Msdnet_b2_step3_scale1_normal_convbottleneck_weights,
'Msdnet/b2/step3/scale1/normal/convbottleneck/BatchNorm/beta':Msdnet_b2_step3_scale1_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b2/step3/scale1/normal/convbottleneck/BatchNorm/gamma':Msdnet_b2_step3_scale1_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b2/step3/scale1/normal/convnormal/weights':Msdnet_b2_step3_scale1_normal_convnormal_weights,
'Msdnet/b2/step3/scale1/normal/convnormal/BatchNorm/beta':Msdnet_b2_step3_scale1_normal_convnormal_BatchNorm_beta,
'Msdnet/b2/step3/scale1/normal/convnormal/BatchNorm/gamma':Msdnet_b2_step3_scale1_normal_convnormal_BatchNorm_gamma,
'Msdnet/b3/step0/scale0/normal/convbottleneck/weights':Msdnet_b3_step0_scale0_normal_convbottleneck_weights,
'Msdnet/b3/step0/scale0/normal/convbottleneck/BatchNorm/beta':Msdnet_b3_step0_scale0_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b3/step0/scale0/normal/convbottleneck/BatchNorm/gamma':Msdnet_b3_step0_scale0_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b3/step0/scale0/normal/convnormal/weights':Msdnet_b3_step0_scale0_normal_convnormal_weights,
'Msdnet/b3/step0/scale0/normal/convnormal/BatchNorm/beta':Msdnet_b3_step0_scale0_normal_convnormal_BatchNorm_beta,
'Msdnet/b3/step0/scale0/normal/convnormal/BatchNorm/gamma':Msdnet_b3_step0_scale0_normal_convnormal_BatchNorm_gamma,
'Msdnet/b3/step0/scale1/down/convbottleneck/weights':Msdnet_b3_step0_scale1_down_convbottleneck_weights,
'Msdnet/b3/step0/scale1/down/convbottleneck/BatchNorm/beta':Msdnet_b3_step0_scale1_down_convbottleneck_BatchNorm_beta,
'Msdnet/b3/step0/scale1/down/convbottleneck/BatchNorm/gamma':Msdnet_b3_step0_scale1_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b3/step0/scale1/down/convdowm/weights':Msdnet_b3_step0_scale1_down_convdowm_weights,
'Msdnet/b3/step0/scale1/down/convdowm/BatchNorm/beta':Msdnet_b3_step0_scale1_down_convdowm_BatchNorm_beta,
'Msdnet/b3/step0/scale1/down/convdowm/BatchNorm/gamma':Msdnet_b3_step0_scale1_down_convdowm_BatchNorm_gamma,
'Msdnet/b3/step0/scale1/normal/convbottleneck/weights':Msdnet_b3_step0_scale1_normal_convbottleneck_weights,
'Msdnet/b3/step0/scale1/normal/convbottleneck/BatchNorm/beta':Msdnet_b3_step0_scale1_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b3/step0/scale1/normal/convbottleneck/BatchNorm/gamma':Msdnet_b3_step0_scale1_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b3/step0/scale1/normal/convnormal/weights':Msdnet_b3_step0_scale1_normal_convnormal_weights,
'Msdnet/b3/step0/scale1/normal/convnormal/BatchNorm/beta':Msdnet_b3_step0_scale1_normal_convnormal_BatchNorm_beta,
'Msdnet/b3/step0/scale1/normal/convnormal/BatchNorm/gamma':Msdnet_b3_step0_scale1_normal_convnormal_BatchNorm_gamma,
'Msdnet/b3/step1/scale0/normal/convbottleneck/weights':Msdnet_b3_step1_scale0_normal_convbottleneck_weights,
'Msdnet/b3/step1/scale0/normal/convbottleneck/BatchNorm/beta':Msdnet_b3_step1_scale0_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b3/step1/scale0/normal/convbottleneck/BatchNorm/gamma':Msdnet_b3_step1_scale0_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b3/step1/scale0/normal/convnormal/weights':Msdnet_b3_step1_scale0_normal_convnormal_weights,
'Msdnet/b3/step1/scale0/normal/convnormal/BatchNorm/beta':Msdnet_b3_step1_scale0_normal_convnormal_BatchNorm_beta,
'Msdnet/b3/step1/scale0/normal/convnormal/BatchNorm/gamma':Msdnet_b3_step1_scale0_normal_convnormal_BatchNorm_gamma,
'Msdnet/b3/step1/scale1/down/convbottleneck/weights':Msdnet_b3_step1_scale1_down_convbottleneck_weights,
'Msdnet/b3/step1/scale1/down/convbottleneck/BatchNorm/beta':Msdnet_b3_step1_scale1_down_convbottleneck_BatchNorm_beta,
'Msdnet/b3/step1/scale1/down/convbottleneck/BatchNorm/gamma':Msdnet_b3_step1_scale1_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b3/step1/scale1/down/convdowm/weights':Msdnet_b3_step1_scale1_down_convdowm_weights,
'Msdnet/b3/step1/scale1/down/convdowm/BatchNorm/beta':Msdnet_b3_step1_scale1_down_convdowm_BatchNorm_beta,
'Msdnet/b3/step1/scale1/down/convdowm/BatchNorm/gamma':Msdnet_b3_step1_scale1_down_convdowm_BatchNorm_gamma,
'Msdnet/b3/step1/scale1/normal/convbottleneck/weights':Msdnet_b3_step1_scale1_normal_convbottleneck_weights,
'Msdnet/b3/step1/scale1/normal/convbottleneck/BatchNorm/beta':Msdnet_b3_step1_scale1_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b3/step1/scale1/normal/convbottleneck/BatchNorm/gamma':Msdnet_b3_step1_scale1_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b3/step1/scale1/normal/convnormal/weights':Msdnet_b3_step1_scale1_normal_convnormal_weights,
'Msdnet/b3/step1/scale1/normal/convnormal/BatchNorm/beta':Msdnet_b3_step1_scale1_normal_convnormal_BatchNorm_beta,
'Msdnet/b3/step1/scale1/normal/convnormal/BatchNorm/gamma':Msdnet_b3_step1_scale1_normal_convnormal_BatchNorm_gamma,
'Msdnet/b3/step2/scale0/normal/convbottleneck/weights':Msdnet_b3_step2_scale0_normal_convbottleneck_weights,
'Msdnet/b3/step2/scale0/normal/convbottleneck/BatchNorm/beta':Msdnet_b3_step2_scale0_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b3/step2/scale0/normal/convbottleneck/BatchNorm/gamma':Msdnet_b3_step2_scale0_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b3/step2/scale0/normal/convnormal/weights':Msdnet_b3_step2_scale0_normal_convnormal_weights,
'Msdnet/b3/step2/scale0/normal/convnormal/BatchNorm/beta':Msdnet_b3_step2_scale0_normal_convnormal_BatchNorm_beta,
'Msdnet/b3/step2/scale0/normal/convnormal/BatchNorm/gamma':Msdnet_b3_step2_scale0_normal_convnormal_BatchNorm_gamma,
'Msdnet/b3/step2/scale1/down/convbottleneck/weights':Msdnet_b3_step2_scale1_down_convbottleneck_weights,
'Msdnet/b3/step2/scale1/down/convbottleneck/BatchNorm/beta':Msdnet_b3_step2_scale1_down_convbottleneck_BatchNorm_beta,
'Msdnet/b3/step2/scale1/down/convbottleneck/BatchNorm/gamma':Msdnet_b3_step2_scale1_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b3/step2/scale1/down/convdowm/weights':Msdnet_b3_step2_scale1_down_convdowm_weights,
'Msdnet/b3/step2/scale1/down/convdowm/BatchNorm/beta':Msdnet_b3_step2_scale1_down_convdowm_BatchNorm_beta,
'Msdnet/b3/step2/scale1/down/convdowm/BatchNorm/gamma':Msdnet_b3_step2_scale1_down_convdowm_BatchNorm_gamma,
'Msdnet/b3/step2/scale1/normal/convbottleneck/weights':Msdnet_b3_step2_scale1_normal_convbottleneck_weights,
'Msdnet/b3/step2/scale1/normal/convbottleneck/BatchNorm/beta':Msdnet_b3_step2_scale1_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b3/step2/scale1/normal/convbottleneck/BatchNorm/gamma':Msdnet_b3_step2_scale1_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b3/step2/scale1/normal/convnormal/weights':Msdnet_b3_step2_scale1_normal_convnormal_weights,
'Msdnet/b3/step2/scale1/normal/convnormal/BatchNorm/beta':Msdnet_b3_step2_scale1_normal_convnormal_BatchNorm_beta,
'Msdnet/b3/step2/scale1/normal/convnormal/BatchNorm/gamma':Msdnet_b3_step2_scale1_normal_convnormal_BatchNorm_gamma,
'Msdnet/b3/step3/scale0/down/convbottleneck/weights':Msdnet_b3_step3_scale0_down_convbottleneck_weights,
'Msdnet/b3/step3/scale0/down/convbottleneck/BatchNorm/beta':Msdnet_b3_step3_scale0_down_convbottleneck_BatchNorm_beta,
'Msdnet/b3/step3/scale0/down/convbottleneck/BatchNorm/gamma':Msdnet_b3_step3_scale0_down_convbottleneck_BatchNorm_gamma,
'Msdnet/b3/step3/scale0/down/convdowm/weights':Msdnet_b3_step3_scale0_down_convdowm_weights,
'Msdnet/b3/step3/scale0/down/convdowm/BatchNorm/beta':Msdnet_b3_step3_scale0_down_convdowm_BatchNorm_beta,
'Msdnet/b3/step3/scale0/down/convdowm/BatchNorm/gamma':Msdnet_b3_step3_scale0_down_convdowm_BatchNorm_gamma,
'Msdnet/b3/step3/scale0/normal/convbottleneck/weights':Msdnet_b3_step3_scale0_normal_convbottleneck_weights,
'Msdnet/b3/step3/scale0/normal/convbottleneck/BatchNorm/beta':Msdnet_b3_step3_scale0_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b3/step3/scale0/normal/convbottleneck/BatchNorm/gamma':Msdnet_b3_step3_scale0_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b3/step3/scale0/normal/convnormal/weights':Msdnet_b3_step3_scale0_normal_convnormal_weights,
'Msdnet/b3/step3/scale0/normal/convnormal/BatchNorm/beta':Msdnet_b3_step3_scale0_normal_convnormal_BatchNorm_beta,
'Msdnet/b3/step3/scale0/normal/convnormal/BatchNorm/gamma':Msdnet_b3_step3_scale0_normal_convnormal_BatchNorm_gamma,
'Msdnet/b3/transition/scale0/conv1/weights':Msdnet_b3_transition_scale0_conv1_weights,
'Msdnet/b3/transition/scale0/conv1/BatchNorm/beta':Msdnet_b3_transition_scale0_conv1_BatchNorm_beta,
'Msdnet/b3/transition/scale0/conv1/BatchNorm/gamma':Msdnet_b3_transition_scale0_conv1_BatchNorm_gamma,
'Msdnet/b4/step0/scale0/normal/convbottleneck/weights':Msdnet_b4_step0_scale0_normal_convbottleneck_weights,
'Msdnet/b4/step0/scale0/normal/convbottleneck/BatchNorm/beta':Msdnet_b4_step0_scale0_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b4/step0/scale0/normal/convbottleneck/BatchNorm/gamma':Msdnet_b4_step0_scale0_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b4/step0/scale0/normal/convnormal/weights':Msdnet_b4_step0_scale0_normal_convnormal_weights,
'Msdnet/b4/step0/scale0/normal/convnormal/BatchNorm/beta':Msdnet_b4_step0_scale0_normal_convnormal_BatchNorm_beta,
'Msdnet/b4/step0/scale0/normal/convnormal/BatchNorm/gamma':Msdnet_b4_step0_scale0_normal_convnormal_BatchNorm_gamma,
'Msdnet/b4/step1/scale0/normal/convbottleneck/weights':Msdnet_b4_step1_scale0_normal_convbottleneck_weights,
'Msdnet/b4/step1/scale0/normal/convbottleneck/BatchNorm/beta':Msdnet_b4_step1_scale0_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b4/step1/scale0/normal/convbottleneck/BatchNorm/gamma':Msdnet_b4_step1_scale0_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b4/step1/scale0/normal/convnormal/weights':Msdnet_b4_step1_scale0_normal_convnormal_weights,
'Msdnet/b4/step1/scale0/normal/convnormal/BatchNorm/beta':Msdnet_b4_step1_scale0_normal_convnormal_BatchNorm_beta,
'Msdnet/b4/step1/scale0/normal/convnormal/BatchNorm/gamma':Msdnet_b4_step1_scale0_normal_convnormal_BatchNorm_gamma,
'Msdnet/b4/step2/scale0/normal/convbottleneck/weights':Msdnet_b4_step2_scale0_normal_convbottleneck_weights,
'Msdnet/b4/step2/scale0/normal/convbottleneck/BatchNorm/beta':Msdnet_b4_step2_scale0_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b4/step2/scale0/normal/convbottleneck/BatchNorm/gamma':Msdnet_b4_step2_scale0_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b4/step2/scale0/normal/convnormal/weights':Msdnet_b4_step2_scale0_normal_convnormal_weights,
'Msdnet/b4/step2/scale0/normal/convnormal/BatchNorm/beta':Msdnet_b4_step2_scale0_normal_convnormal_BatchNorm_beta,
'Msdnet/b4/step2/scale0/normal/convnormal/BatchNorm/gamma':Msdnet_b4_step2_scale0_normal_convnormal_BatchNorm_gamma,
'Msdnet/b4/step3/scale0/normal/convbottleneck/weights':Msdnet_b4_step3_scale0_normal_convbottleneck_weights,
'Msdnet/b4/step3/scale0/normal/convbottleneck/BatchNorm/beta':Msdnet_b4_step3_scale0_normal_convbottleneck_BatchNorm_beta,
'Msdnet/b4/step3/scale0/normal/convbottleneck/BatchNorm/gamma':Msdnet_b4_step3_scale0_normal_convbottleneck_BatchNorm_gamma,
'Msdnet/b4/step3/scale0/normal/convnormal/weights':Msdnet_b4_step3_scale0_normal_convnormal_weights,
'Msdnet/b4/step3/scale0/normal/convnormal/BatchNorm/beta':Msdnet_b4_step3_scale0_normal_convnormal_BatchNorm_beta,
'Msdnet/b4/step3/scale0/normal/convnormal/BatchNorm/gamma':Msdnet_b4_step3_scale0_normal_convnormal_BatchNorm_gamma,
}

model_weights = {}
for k, v in model_weights_temp.items():
    if len(v.shape) == 4:
        model_weights[k] = np.transpose(v, (2, 3, 1, 0))
    elif len(v.shape) == 2:
        model_weights[k] = np.transpose(v)
    else:
        model_weights[k] = v


with tf.Graph().as_default():
    #global_step = tf.Variable(0, trainable=False, name='global_step')
        images = tf.placeholder(tf.float32, shape=(32, 448, 448, 3))
        #images = [tf.placeholder(tf.float32, [2, 224, 224, 3])]
        #labels = [tf.placeholder(tf.int32, [2])]
        
        #network_train = resnet.ResNet(hp, images, labels, global_step, name="train")
        #network_train.build_model()
        channels = 50
        dets = inference(images, channels,is_training=True)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(
                visible_device_list='1',
                per_process_gpu_memory_fraction=0.8)
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options = gpu_options,
            allow_soft_placement=True,
            log_device_placement=False))
        sess.run(init)

        # Set variables values
        print('Set variables to loaded weights')
        all_vars = tf.trainable_variables()
        for v in all_vars:
            print('\t' + v.op.name)
            #print (v.shape,model_weights[v.op.name].shape)
            if v.op.name =='convbox_detection/Conv/weights' or v.op.name =='convbox_detection/Conv/biases':
               continue
            assert(v.shape==model_weights[v.op.name].shape),'shape dismatch'
            assign_op = v.assign(model_weights[v.op.name])
            sess.run(assign_op)

        # Save as checkpoint
        print('Save as checkpoint: %s' % INIT_CHECKPOINT_DIR)
        if not os.path.exists(INIT_CHECKPOINT_DIR):
            os.mkdir(INIT_CHECKPOINT_DIR)
        saver = tf.train.Saver(tf.global_variables())
        saver.save(sess, os.path.join(INIT_CHECKPOINT_DIR, 'exportmodel.ckpt'))

        print('Done!')
