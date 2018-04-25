# --------------------------------------------------------
# ConvBox
# Copyright (c) 2016 HUST
# Licensed under The MIT License [see LICENSE for details]
# Written by Ckb
# --------------------------------------------------------

import re
import gflags
import numpy as np

import tensorflow as tf
from nets.interface import basenet
from nets.interface import basenet_arg_scope
from nets.interface import output2X
from convbox_loss import convbox_loss
from convbox_loss import convbox_loss2

slim = tf.contrib.slim

FLAGS = gflags.FLAGS
# Basic model parameters.
gflags.DEFINE_enum('basenet', 'Msdnet', ['InceptionV2', 'InceptionV3', 'InceptionV4','Msdnet'],
                            """Name of base network.""")

 
def inference(images, channels,is_training=False, scope=None):
    with tf.name_scope(scope, 'tower', [images]):
        with slim.arg_scope(basenet_arg_scope('Msdnet')):
            with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
                net, end_points = basenet(images, 'Msdnet')
                print (net)
                #net = slim.conv2d(net, 1536, 1, stride=1,padding='SAME', scope='final_conv1')
                #net = slim.conv2d(net, 1536, 3, stride=1,padding='SAME', scope='final_conv2')
                #net = slim.conv2d(net, 1536, 3, stride=1,padding='SAME', scope='final_conv3')
                #net = slim.conv2d(net, 50, 3, stride=1,padding='SAME', scope='final_conv4')
                
                # add detection layers
                with tf.variable_scope('convbox_detection'):
                    with slim.arg_scope([slim.conv2d], stride=1, padding='SAME',
                            weights_initializer = tf.truncated_normal_initializer(stddev=0.01),
                            normalizer_fn=None, normalizer_params=None):
                            net = slim.conv2d(net, channels, [1, 1], activation_fn=None)
                            print (net)
                            det_branch = [tf.sigmoid(net[:, :, :, :2])]
                            det_branch.append(tf.nn.softmax(net[:, :, :, 2:22]))
                            det_branch.append(tf.nn.softmax(net[:, :, :, 22:42]))
                            det_branch.append(net[:, :, :, 42:])
                            det_branch = tf.concat(det_branch, 3)
                '''
                with tf.variable_scope('convbox_detection'):
                    with slim.arg_scope([slim.conv2d], stride=1, padding='SAME',
                            weights_initializer = tf.truncated_normal_initializer(stddev=0.01),
                            normalizer_fn=None, normalizer_params=None):
                        
                        
                        
                        Mixed_2X = output2X(FLAGS.basenet)[0]
                        Mixed_2X_C1 = output2X(FLAGS.basenet)[1]
                        Mixed_2X_C2 = output2X(FLAGS.basenet)[2]
                        
                        if FLAGS.basenet == 'InceptionV2':
                            net_28 = tf.reshape(end_points[Mixed_2X], [-1, 2*FLAGS.output_row, FLAGS.output_col, 2*Mixed_2X_C1])
                            net_28 = tf.transpose(net_28, perm=[0, 2, 1, 3])
                            net_28 = tf.reshape(net_28, [-1, FLAGS.output_col, FLAGS.output_row, 4*Mixed_2X_C1])
                            net_28 = tf.transpose(net_28, perm=[0, 2, 1, 3])
                            net = tf.concat([net_28, net], 3)
                        else:
                            net_29 = slim.max_pool2d(end_points[Mixed_2X], [3, 3], stride=2, padding='VALID')
                            net = tf.concat([net_29, net], 3)
                        
                        #print (net)
                        #aa=raw_input('pause')
                        
                        net = slim.conv2d(net, 1536, [1, 1])
                        net = slim.conv2d(net, 1536, [3, 3])
                        net = slim.conv2d(net, 1536, [3, 3])
                        
                        #print (net)
                        #aa=raw_input('pause')
                        ss = [(-1, -1), (+1, -1), (-1, +1), (+1, +1)]
                        branch_list = ['det_lt', 'det_rt', 'det_lb', 'det_rb']
                        for k, branch in enumerate(branch_list):
                            net_branch = slim.conv2d(net, 1536, [3, 3], rate=2)
                            #print net_branch.get_shape()
                            net_branch = slim.conv2d(net_branch, channels, [3, 3], rate=2)
                            net_branch = net_branch + slim.conv2d(net_branch, channels, [3, 3], rate=4)
                            net_branch = net_branch + slim.conv2d(net_branch, channels, [3, 3], rate=8)
                            net_branch = net_branch + slim.conv2d(net_branch, channels, [3, 3], rate=16)
                            net_branch = net_branch + slim.conv2d(net_branch, channels, [3, 3], rate=1)
                            net_branch = net_branch + slim.conv2d(net_branch, channels, [1, 1], rate=1, activation_fn=None)
                            
                            offset1 = FLAGS.num_box * 2
                            offset2 = FLAGS.num_box * 6
                            offset3 = FLAGS.num_box * (6 + 2*(FLAGS.output_row+FLAGS.output_col))
                            step21 = FLAGS.output_row
                            step22 = FLAGS.output_col
                            step3 = FLAGS.num_class
                            
                            det_branch = [tf.sigmoid(net_branch[:, :, :, :offset1])]
                            det_branch.append(net_branch[:, :, :, offset1:offset2])                            
                            for i in xrange(2*FLAGS.num_box):
                                det_branch.append(tf.nn.softmax(net_branch[:, :, :, offset2+i*(step21+step22):offset2+i*(step21+step22)+step21]))
                                det_branch.append(tf.nn.softmax(net_branch[:, :, :, offset2+i*(step21+step22)+step21:offset2+(i+1)*(step21+step22)]))
                            for i in xrange(2*FLAGS.num_box):
                                det_branch.append(tf.nn.softmax(net_branch[:, :, :, offset3+i*step3:offset3+(i+1)*step3]))
                            det_branch = tf.concat(det_branch, 3)
                            print (det_branch)
                            end_points[branch] = det_branch
                            '''                        
    activation_summaries(end_points) 
    #return [end_points[x] for x in branch_list]
    return det_branch
    #return net

    
def tower_loss(images, labels, scope,reuse_variables=None):
    channels = FLAGS.num_box * labels[0].get_shape().as_list()[-1]
    print (channels)
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        dets = inference(images, channels, is_training=True, scope=scope)
    print ('dets',dets)
    print ('labels',labels)
    #loss = [convbox_loss(dets[i], labels[i]) for i in xrange(4)]
    #loss = [convbox_loss(dets[i], labels) for i in xrange(10)]
    loss = convbox_loss(dets,labels)
    print (1)
    #loss = convbox_loss(dets, labels)
    losses = tf.losses.get_losses(scope=scope)
    regularization_losses = tf.losses.get_regularization_losses()
    total_loss = tf.add_n(losses + regularization_losses, name='total_loss')
    print (2)
    # compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [loss] + [total_loss])

    # attach a scalar summmary to all individual losses and the total loss.
    for l in losses + [loss] + [total_loss]:
        # remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. this helps the clarity of presentation on TensorBoard.
        loss_name = re.sub('tower_[0-9]*/', '', l.op.name)
        # name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(loss_name +' (raw)', l)
        tf.summary.scalar(loss_name, loss_averages.average(l))

    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss
    
def tower_loss2(images, labels, scope, reuse_variables=None):
    channels = FLAGS.num_box * labels[0].get_shape().as_list()[-1]
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        dets = inference(images, channels, is_training=True, scope=scope)
    invalid_mask = generate_invalid_mask()
    loss = [convbox_loss2(dets[i], labels[i], invalid_mask[i]) for i in xrange(4)]
    losses = tf.losses.get_losses(scope=scope)
    regularization_losses = tf.losses.get_regularization_losses()
    total_loss = tf.add_n(losses + regularization_losses, name='total_loss')

    # compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + loss + [total_loss])

    # attach a scalar summmary to all individual losses and the total loss.
    for l in losses + loss + [total_loss]:
        # remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. this helps the clarity of presentation on TensorBoard.
        loss_name = re.sub('tower_[0-9]*/', '', l.op.name)
        # name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(loss_name +' (raw)', l)
        tf.summary.scalar(loss_name, loss_averages.average(l))

    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss

def generate_invalid_mask():
    ss = [(-1, -1), (+1, -1), (-1, +1), (+1, +1)]
    channels = 2*FLAGS.num_box*(FLAGS.output_row + FLAGS.output_col)
    invalid_mask = np.ones((4, FLAGS.output_row, FLAGS.output_col, channels), dtype=bool)
    for k in xrange(4):
        for i in xrange(FLAGS.output_row):
            for j in xrange(FLAGS.output_col):
                p_mc_start, p_mc_end = get_range_mc(i, FLAGS.output_row, ss[k][1])
                q_mc_start, q_mc_end = get_range_mc(j, FLAGS.output_col, ss[k][0])
                p_cm_start, p_cm_end = get_range_cm(i, FLAGS.output_row, ss[k][1])
                q_cm_start, q_cm_end = get_range_cm(j, FLAGS.output_col, ss[k][0])
                for n in xrange(FLAGS.num_box):
                    offset0 = n*(FLAGS.output_row + FLAGS.output_col)
                    offset1 = offset0 + FLAGS.output_row
                    offset2 = (FLAGS.num_box + n)*(FLAGS.output_row + FLAGS.output_col)
                    offset3 = offset2 + FLAGS.output_row
                    invalid_mask[k, i, j, p_mc_start+offset0:p_mc_end+offset0] = False
                    invalid_mask[k, i, j, q_mc_start+offset1:q_mc_end+offset1] = False
                    invalid_mask[k, i, j, p_cm_start+offset2:p_cm_end+offset2] = False
                    invalid_mask[k, i, j, q_cm_start+offset3:q_cm_end+offset3] = False
                    
    return tf.convert_to_tensor(invalid_mask)
                
                
def get_range_mc(i, rows, ss):
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

def get_range_cm(i, rows, ss):
    if ss < 0:
        return i, int((i + rows)/2) + 1
    else:
        return int(i/2), i + 1

def activation_summary(x):
    # remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('tower_[0-9]*/', '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def activation_summaries(endpoints):
    with tf.name_scope('summaries'):
        for act in endpoints.values():
            activation_summary(act)
