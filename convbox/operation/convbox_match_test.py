# --------------------------------------------------------
# ConvBox
# Copyright (c) 2016 HUST
# Licensed under The MIT License [see LICENSE for details]
# Written by Ckb
# --------------------------------------------------------

import numpy as np
import tensorflow as tf
from convbox_match_op import convbox_match
import sys
def l2_loss(tensor, weight=1.0, scope=None):
    with tf.name_scope(scope, 'l2_loss', [tensor]):
        weight = tf.convert_to_tensor(weight,
                                      dtype=tensor.dtype.base_dtype,
                                      name='loss_weight')
        loss = tf.multiply(weight, tf.nn.l2_loss(tensor), name='value')
        tf.losses.add_loss(loss)
        return loss
        
det = tf.constant([[[[0.5,0.1, 0.5,0,0,0, 0,0,0,0, 0.3,0.3,0.8,0.8 ], 
                     [0,  0, 0,0,0,0, 0,  0,  0,  0,   0,0,0,0]], 
                    [[0,  0, 0,0,0,0, 0,  0,  0,  0,   0,0,0,0], 
                     [0,  0, 0,0,0,0, 0,  0,  0,  0,   0,0,0,0]]]])
labels = tf.constant([[[[1, 1,0, 0.4,0.4,0.7,0.7], 
                        [0,0,0,0,0,0,0]], 
                       [[0,0,0,0,0,0,0], 
                        [0,0,0,0,0,0,0]]]])

det_mask1 = np.array([[[[1,0, 1,1,0,0, 1,1,1,1, 0,0,0,0], 
                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0]], 
                      [[0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0]]]])
labels_mask1 = np.array([[[[1,1,1,1,1,1,1], 
                          [0,0,0,0,0,0,0]], 
                         [[0,0,0,0,0,0,0], 
                          [0,0,0,0,0,0,0]]]])

det_mask, label_mask, update_label = convbox_match(det, labels, 2, 2)

mask = np.array([[[[ True, False, False, False,  True,  True, False, False, False, False, False,False,  True,  True,  True,  True, False, False, False, False, False, False,
    False, False, False, False, False, False,  False,  False, False, False, False,
    False, False, False],
   [True, False,  True, False, False, False, False, False,  True,  True, False,
    False, False, False, False, False, False, False, False, False,  True,  True,
     True,  True, False, False, False, False, False, False, False, False,  True,
     True, False, False]],
  [[True, False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False, False,
    False, False, False],
   [False, False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False, False,
    False, True, True]]]])

#batch_size = det.get_shape().as_list()[0]
#object_scale = FLAGS.object_scale * weight / float(batch_size)
#noobject_scale = FLAGS.noobject_scale * weight / float(batch_size)
#coord_scale = FLAGS.coord_scale * weight / float(batch_size)

#connection_scale = FLAGS.connection_scale * weight / float(batch_size)

#class_scale = FLAGS.class_scale * weight / float(batch_size)
with tf.Session() as sess:
    #print det
    #print labels
    #det = sess.run(det)
    det_mask,label_mask,update_label = sess.run([det_mask, label_mask, update_label])
    noobj = tf.boolean_mask(det[:, :, :, :2],
                                (tf.logical_not(det_mask[:, :, :, :2])))
                                
    print (noobj)
    l1 = l2_loss(noobj, weight=1.0, scope='noobject_scale')
    
    obj_1 = tf.boolean_mask(det[:, :, :, :2],
                                det_mask[:, :, :, :2])
    
    obj_2 = tf.boolean_mask(update_label[:, :, :, :1],
                                label_mask[:, :, :, :1])
    print (obj_1)
    print (obj_2)
    l2 = l2_loss(obj_1 - obj_2, weight=1.0, scope='object_scale')
    
    coord_1 = tf.boolean_mask(det[:, :, :, 2*(2+1):],
                                  det_mask[:, :, :, 2*(2+1):])
    coord_2 = tf.boolean_mask(update_label[:, :, :, 3:],
                              label_mask[:, :, :, 3:])
    l3 = l2_loss(coord_1 - coord_2, weight=1.0, scope='coord_scale')
    
    cls_1 = tf.boolean_mask(det[:, :, :, 2:2*(2+1)],
                                det_mask[:, :, :, 2:2*(2+1)])
    cls_2 = tf.boolean_mask(update_label[:, :, :, 1:3],
                            label_mask[:, :, :, 1:3])
    l5 = l2_loss(cls_1 - cls_2, weight=1.0, scope='class_scale')
    loss1, loss2, loss3, loss5 = sess.run([l1, l2, l3,l5])
    #obj = tf.boolean_mask(det, mask)
    #n = sess.run(noobj)
    #print n
    print (loss1,loss2,loss3,loss5)
    
    #print mask2
    #if (mask1 == det_mask).all() and (mask2 == labels_mask).all():
    #    print 'pass'
    #else:
    #    print 'fail'

        
