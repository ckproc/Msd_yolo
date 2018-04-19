# --------------------------------------------------------
# ConvBox
# Copyright (c) 2016 HUST
# Licensed under The MIT License [see LICENSE for details]
# Written by Ckb
# --------------------------------------------------------

import gflags
import tensorflow as tf
from tensorflow.python.framework import ops
from operation.convbox_match_op import convbox_match


FLAGS = gflags.FLAGS
# Basic model parameters.
gflags.DEFINE_float('object_scale', 1.0,
                          """Object scale of convbox loss.""")
gflags.DEFINE_float('noobject_scale', 0.3,
                          """Noobject scale of convbox loss.""")
gflags.DEFINE_float('connection_scale', 1.0,
                          """Class scale of convbox loss.""")
gflags.DEFINE_float('coord_scale', 5.0,
                          """Coord scale of convbox loss.""")
gflags.DEFINE_float('class_scale', 1.0,
                          """Class scale of convbox loss.""")
gflags.DEFINE_integer('num_box', 2,
                            """Number of box of convbox loss.""")

@ops.RegisterGradient("ConvboxMatch")
def _convbox_match_grad(op, grad1, grad2, grad3):
    """
    The gradients for `convbox_match`.
    """
    return None, None, None, None

@ops.RegisterShape("ConvboxMatch")
def _convbox_match_shape(op):
    """
    Shape function for the ConvboxMatch op.
    This is the unconstrained version of ZeroOut, which produces an output
    with the same shape as its input.
    """
    return op.inputs[0].get_shape(), op.inputs[1].get_shape(), op.inputs[1].get_shape()


def convbox_loss(det, label, weight=1.0, scope=None):
    '''
    det[batch,14,14,204]
    label[batch,14,14,102]
    '''
    with tf.name_scope(scope, 'convbox_loss', [det, label]):

        batch_size = det.get_shape().as_list()[0]
        print ("bs:",batch_size)
        object_scale = FLAGS.object_scale * weight / float(batch_size)
        noobject_scale = FLAGS.noobject_scale * weight / float(batch_size)
        coord_scale = FLAGS.coord_scale * weight / float(batch_size)
        
        connection_scale = FLAGS.connection_scale * weight / float(batch_size)
        
        class_scale = FLAGS.class_scale * weight / float(batch_size)
        
        num_box = FLAGS.num_box
        num_class = FLAGS.num_class
        output_row = FLAGS.output_row
        output_col = FLAGS.output_col
        print (object_scale,noobject_scale,coord_scale,class_scale)
        print (num_box,num_class,output_row,output_col)
        
        #det[:,:,:,:num_box] = tf.sigmoid(det[:, :, :, :num_box])
        det_mask, label_mask, update_label = convbox_match(det, label, num_box, num_class)
        
        #print (update_label)
        noobj = tf.boolean_mask(det[:, :, :, :num_box],
                                (tf.logical_not(det_mask[:, :, :, :num_box])))
        l1 = l2_loss(noobj, weight=noobject_scale, scope='noobject_scale')
        
        obj_1 = tf.boolean_mask(det[:, :, :, :num_box],
                                det_mask[:, :, :, :num_box])
        obj_2 = tf.boolean_mask(update_label[:, :, :, :1],
                                label_mask[:, :, :, :1])
 
        l2 = l2_loss(obj_1 - obj_2, weight=object_scale, scope='object_scale')
        
        coord_1 = tf.boolean_mask(det[:, :, :, 2*(num_class+1):],
                                  det_mask[:, :, :, 2*(num_class+1):])
        coord_2 = tf.boolean_mask(update_label[:, :, :, 21:],
                                  label_mask[:, :, :, 21:])
        l3 = l2_loss(coord_1 - coord_2, weight=coord_scale, scope='coord_scale')

        #conn_1 = tf.boolean_mask(det[:, :, :, 6*num_box:(6+2*(output_row+output_col))*num_box],
        #                         det_mask[:, :, :, 6*num_box:(6+2*(output_row+output_col))*num_box])
        #conn_2 = tf.boolean_mask(update_label[:, :, :, 6:6+2*(output_row+output_col)],
        #                         label_mask[:, :, :, 6:6+2*(output_row+output_col)])
        #l4 = l2_loss(conn_1 - conn_2, weight=connection_scale, scope='connection_scale')
        
        cls_1 = tf.boolean_mask(det[:, :, :, num_box:num_box*(num_class+1)],
                                det_mask[:, :, :, num_box:num_box*(num_class+1)])
        cls_2 = tf.boolean_mask(update_label[:, :, :, 1:21],
                                label_mask[:, :, :, 1:21])
        l5 = l2_loss(cls_1 - cls_2, weight=class_scale, scope='class_scale')
        
        loss =l1+ l2 + l3 + l5
        return loss

def convbox_loss2(det, label, invalid_mask, weight=1.0, scope=None):
    with tf.name_scope(scope, 'convbox_loss', [det, label]):

        batch_size = det.get_shape().as_list()[0]
        object_scale = FLAGS.object_scale * weight / batch_size
        noobject_scale = FLAGS.noobject_scale * weight / batch_size
        coord_scale = FLAGS.coord_scale * weight / batch_size
        connection_scale = FLAGS.connection_scale * weight / batch_size
        class_scale = FLAGS.class_scale * weight / batch_size
        invalid_scale = 0.0 * weight / batch_size
        num_box = FLAGS.num_box
        num_class = FLAGS.num_class
        output_row = FLAGS.output_row
        output_col = FLAGS.output_col
        
        det_mask, label_mask, update_label = convbox_match(det, label, num_box, num_class)
        
        noobj = tf.boolean_mask(det[:, :, :, :2*num_box],
                                tf.logical_not(det_mask[:, :, :, :2*num_box]))
        l1 = l2_loss(noobj, weight=noobject_scale, scope='noobject_scale')
        
        obj_1 = tf.boolean_mask(det[:, :, :, :2*num_box],
                                det_mask[:, :, :, :2*num_box])
        obj_2 = tf.boolean_mask(update_label[:, :, :, :2],
                                label_mask[:, :, :, :2])
        l2 = l2_loss(obj_1 - obj_2, weight=object_scale, scope='object_scale')
        
        coord_1 = tf.boolean_mask(det[:, :, :, 2*num_box:6*num_box],
                                  det_mask[:, :, :, 2*num_box:6*num_box])
        coord_2 = tf.boolean_mask(update_label[:, :, :, 2:6],
                                  label_mask[:, :, :, 2:6])
        l3 = l2_loss(coord_1 - coord_2, weight=coord_scale, scope='coord_scale')

        conn_1 = tf.boolean_mask(det[:, :, :, 6*num_box:(6+2*(output_row+output_col))*num_box],
                                 det_mask[:, :, :, 6*num_box:(6+2*(output_row+output_col))*num_box])
        conn_2 = tf.boolean_mask(update_label[:, :, :, 6:6+2*(output_row+output_col)],
                                 label_mask[:, :, :, 6:6+2*(output_row+output_col)])
        l4 = l2_loss(conn_1 - conn_2, weight=connection_scale, scope='connection_scale')
        
        cls_1 = tf.boolean_mask(det[:, :, :, (6+2*(output_row+output_col))*num_box:],
                                det_mask[:, :, :, (6+2*(output_row+output_col))*num_box:])
        cls_2 = tf.boolean_mask(update_label[:, :, :, 6+2*(output_row+output_col):],
                                label_mask[:, :, :, 6+2*(output_row+output_col):])
        l5 = l2_loss(cls_1 - cls_2, weight=class_scale, scope='class_scale')
        
        invalid = []
        for k in xrange(det.get_shape().as_list()[0]):
            invalid.append(tf.boolean_mask(det[k, :, :, 6*num_box:(6+2*(output_row+output_col))*num_box], invalid_mask)) 
        invalid = tf.concat(0, invalid)
        l6 = l2_loss(invalid, weight=invalid_scale, scope='invalid_scale')
        
        loss = l1 + l2 + l3 + l4 + l5 + l6
        return loss
        
def l2_loss(tensor, weight=1.0, scope=None):
    with tf.name_scope(scope, 'l2_loss', [tensor]):
        weight = tf.convert_to_tensor(weight,
                                      dtype=tensor.dtype.base_dtype,
                                      name='loss_weight')
        loss = tf.multiply(weight, tf.nn.l2_loss(tensor), name='value')
        tf.losses.add_loss(loss)
        return loss
