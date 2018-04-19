# --------------------------------------------------------
# ConvBox
# Copyright (c) 2016 HUST
# Licensed under The MIT License [see LICENSE for details]
# Written by Ckb
# --------------------------------------------------------

import sys
import time
import gflags
import os.path
import numpy as np
from datetime import datetime
#import argparse
import tensorflow as tf
from convbox_data import SimpleRunner
from convbox_model import tower_loss
from convbox_model import tower_loss2
import cv2
slim = tf.contrib.slim

FLAGS = gflags.FLAGS
# Basic model parameters.
gflags.DEFINE_enum('mode', 'train', ['train', 'resume'], 
                           """Run mode, train or resume.""")
gflags.DEFINE_string('train_dir', 'backup',
                           """Directory where to write event logs and checkpoint.""")
gflags.DEFINE_enum('optimizer', 'ADAM', ['SGD', 'RMS', 'ADAM'], 
                           """The name of the optimizer, such as SGD or RMS.""")
gflags.DEFINE_string('gpu_list', '0, 1',
                           """GPU list, such as '0, 1, ....'.""")
gflags.DEFINE_float('gpu_usage', 0.8,
                          """Per process gpu memory fraction.""")
gflags.DEFINE_integer('max_steps', 50000,
                            """Number of batches to run.""")
gflags.DEFINE_enum('lr_policy', 'piecewise_constant', ['piecewise_constant'], 
                           """The method to adjust learning rate.""")
gflags.DEFINE_string('step_value', '100, 30000, 40000',
                           """Global step value for setting learning rate.""")
gflags.DEFINE_string('step_lr', '0.001, 0.005, 0.0005, 0.00005',
                           """Learning rate step value.""")
gflags.DEFINE_string('checkpoint_path', 'model/inception-v2/inception_v2.ckpt',
                           """Pretrained model, must be specified.""")


def configure_solver(global_step):
    if FLAGS.lr_policy == 'piecewise_constant':
        boundaries = [int(i) for i in FLAGS.step_value.split(',')]
        values = [float(i) for i in FLAGS.step_lr.split(',')]
        lr = tf.train.piecewise_constant(global_step, boundaries, values)
    else:
        raise ValueError('Lr_policy [%s] was not recognized.', FLAGS.lr_policy)

    # Create an optimizer that performs gradient descent.
    if FLAGS.optimizer == 'SGD':
        opt = tf.train.GradientDescentOptimizer(lr)
        #opt = tf.train.MomentumOptimizer(lr, momentum=0.9, use_nesterov = True)
    elif FLAGS.optimizer == 'RMS':
        print ('RMS::')
        opt = tf.train.RMSPropOptimizer(lr, decay=0.9, momentum=0.9, epsilon=1.0)
    elif FLAGS.optimizer == 'ADAM':
        opt = tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-4)
    else:
        raise ValueError('Optimizer [%s] was not recognized.', FLAGS.optimizer)
    return opt, lr

def model_init(sess):
    if FLAGS.mode == 'train':
        assert tf.gfile.Exists(FLAGS.checkpoint_path)
        variables_to_restore = slim.get_model_variables(scope=FLAGS.basenet)
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, FLAGS.checkpoint_path)
        print '%s: Pre-trained model restored from %s.' %(
                datetime.now(), FLAGS.checkpoint_path)
    elif FLAGS.mode == 'resume':
        restorer = tf.train.Saver(slim.get_variables())
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            restorer.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print '%s: Resume model from %s at step=%s.' %(
                    datetime.now(), ckpt.model_checkpoint_path, global_step)
        else:
            ValueError('No checkpoint found.')
    else:
        raise ValueError('Mode [%s] was not recognized.', FLAGS.mode)

def average_gradients(tower_grads, scope=None):
    """
    Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    """
    with tf.name_scope(scope, 'average_gradients', [tower_grads]):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            # ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)
            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

        
def train():
    
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # configure learning rate and optimizer.
        global_step = tf.get_variable('global_step', [], dtype=tf.int32, 
                initializer=tf.constant_initializer(0), trainable=False)
        opt, lr = configure_solver(global_step)

        print(' build data graph.')
        num_gpus = len(FLAGS.gpu_list.split(','))
        with tf.name_scope('data'):
            simple_runner = SimpleRunner()
            images, labels = simple_runner.get_inputs()
            #print (images)
            #print (labels)
            input_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)

            # split the batch of images and labels for towers.
            assert FLAGS.batch_size % num_gpus == 0, (
                    'Batch size must be divisible by number of GPUs')
            images_splits = tf.split(images, num_gpus, 0)
            labels_splits = tf.split(labels, num_gpus, 0)
        # build net graph.
        tower_grads = []
        reuse_variables = None
        for i in xrange(num_gpus):
            with tf.device('/gpu:%d' % i), tf.name_scope('tower_%d' % i) as scope:
                # force all variables to reside on the CPU.
                with slim.arg_scope([slim.model_variable], device='/cpu:0'):
                    # calculate the loss and share the variables across all towers.
                    loss = tower_loss(images_splits[i], labels_splits[i], scope, reuse_variables)
                # reuse variables for the next tower.
                reuse_variables = True

                # retain the summaries from the final tower.
                summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                # retain the batch normalization updates operations only from the final tower.
                batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)

                # calculate the gradients for the batch of data on this tower.
                grads = opt.compute_gradients(loss)
                tower_grads.append(grads)

        # calculate the mean of each gradient.
        grads = average_gradients(tower_grads)
        # apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # track the moving averages of all model variables.
        variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
        #variables_to_average = (slim.get_model_variables(scope=(FLAGS.basenet+'/Mixed')) 
        #        + slim.get_model_variables(scope='convbox_detection'))
        variables_to_average = tf.trainable_variables()
        variables_averages_op = variable_averages.apply(variables_to_average)

        # group all updates to into a single train op.
        batchnorm_updates_op = tf.group(*batchnorm_updates)
        train_op = tf.group(apply_gradient_op, variables_averages_op, batchnorm_updates_op)
        
        # summaries.
        summaries.extend(input_summaries)
        summaries.append(tf.summary.scalar('learning_rate', lr))
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # merge all summaries.
        summary_op = tf.summary.merge(summaries)
        # create a saver.
        saver = tf.train.Saver(slim.get_variables(), max_to_keep=50)
        # build an initialization operation to run below.
        init = tf.global_variables_initializer()

       
        gpu_options = tf.GPUOptions(
                visible_device_list=FLAGS.gpu_list,
                per_process_gpu_memory_fraction=FLAGS.gpu_usage)
        print ("start running.") 
        sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
                gpu_options=gpu_options))

        sess.run(init)
        # load pretrained model or resume model.
        #model_init(sess)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        #print ("ssss")
        while True:
            start_time = time.time()
            batch_images, batch_labels = simple_runner.get_next_batch_datums()
            duration_data = time.time() - start_time
            feed_dict = {images: batch_images,labels:batch_labels}
            '''
            im,lab =sess.run([images,labels],feed_dict=feed_dict)
            im = im.astype(np.uint8)
            for id in range(len(im)):
                label = lab[id]
                print label.shape
                for i in range(14):
                   for j in range(14):
                      if label[i][j][0]:
                        x,y,w,h = label[i][j][21:]
                        x=(x+j)/14*448
                        #x=x*448
                        y=(y+i)/14*448
                        #y=y*448
                        w=w**2*448
                        #w=w*448
                        #h=h*448
                        h=h**2*448
                        x1=x-w/2
                        y1=y-h/2
                        x2=x+w/2
                        y2=y+h/2
                        cv2.rectangle(im[id], (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
                cv2.imwrite('./images/%d.jpg'%id,im[id])
            
            break
            '''
            #print ("sssss")
            _, loss_value, lr_value, step = sess.run([train_op, loss, lr, global_step], feed_dict=feed_dict)
            #print ("ssssss",step)
            
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            
            if step % 10 == 0:
                examples_per_sec = FLAGS.batch_size / float(duration)
                format_str = ('%s: step %d, lr = %.5f, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch_data; %.3f sec/batch)')
                print format_str % (datetime.now(), step, lr_value, loss_value,
                                    examples_per_sec, duration_data, duration)

            if step % 100 == 0:
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 2000 == 0 or step == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                if step == FLAGS.max_steps:
                    break


def main(unused_argv=None):
    print ('s')
    FLAGS(sys.argv)
    if FLAGS.mode == 'train':
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)
    print ('train()')
    train()
'''
def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument('--max_steps',type=int,help='',default=10000)
    parser.add_argument('--nScales',type=int,help='',default=3)
                
    
    parser.add_argument('--testOnly', 
        help='Run on validation set only.', action='store_true')
    parser.add_argument('--tenCrop', 
        help='Ten-crop testing.', action='store_true')
    parser.add_argument('--reduction',type=float,help='dimension reduction ratio at transition layers',default=0.5)    
    parser.add_argument('--LR', type=float,help='initial learning rate.', default=0.1)
    parser.add_argument('--momentum', type=float, help='.', default=0.9)
    parser.add_argument('--weight_decay', type=float, help='.', default=1e-4)
    parser.add_argument('--base', type=int,
        help='the layer to attach the first classifier', default=4)
        
        
    
    parser.add_argument('--nBlocks', type=int,
        help='number of blocks/classifiers', default=10)
    parser.add_argument('--stepmode', type=str,
        help='patten of span between two adjacent classifers |even|lin_grow|', default='even')
    parser.add_argument('--step', type=int,
        help='span between two adjacent classifers.', default=2)
    parser.add_argument('--bottleneck',
        help='use 1x1 conv layer or not', action='store_true')
    parser.add_argument('--growthRate', type=int,
        help='number of output channels for each layer (the first scale).', default=6)
    parser.add_argument('--grFactor', type=str,
        help='growth rate factor of each sacle', default='1-2-4-4')
    parser.add_argument('--prune', type=str,
        help='specify how to prune the network, min | max', default='max')
    #parser.add_argument('--joinType', type=str,
    #    help='add or concat for features from different paths', default='concat')
    
    parser.add_argument('--bnFactor', type=str,
        help='bottleneck factor of each sacle, 4-4-4-4 | 1-2-4-4', default='1-2-4-4')
    
    
    parser.add_argument('--initChannels',type=int,help='number of features produced by the initial conv layer',default=32)
  
    
    return parser.parse_args(argv)
'''
if __name__ == '__main__':
    
    tf.app.run()
