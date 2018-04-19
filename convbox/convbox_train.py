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

import tensorflow as tf
from convbox_data import CustomRunner
from convbox_model import tower_loss

slim = tf.contrib.slim

FLAGS = gflags.FLAGS
# Basic model parameters.
gflags.DEFINE_enum('mode', 'train', ['train', 'resume'], 
                           """Run mode, train or resume.""")
gflags.DEFINE_string('train_dir', 'backup',
                           """Directory where to write event logs and checkpoint.""")
gflags.DEFINE_enum('optimizer', 'RMS', ['SGD', 'RMS', 'ADAM'], 
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
    elif FLAGS.optimizer == 'RMS':
        opt = tf.train.RMSPropOptimizer(lr, decay=0.9, momentum=0.9, epsilon=1.0)
    elif FLAGS.optimizer == 'ADAM':
        opt = tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1.0)
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

        # build data graph.
        num_gpus = len(FLAGS.gpu_list.split(','))
        with tf.name_scope('data'):
            custom_runner = CustomRunner()
            images, labels = custom_runner.get_inputs()
            input_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)

            # split the batch of images and labels for towers.
            assert FLAGS.batch_size % num_gpus == 0, (
                    'Batch size must be divisible by number of GPUs')
            images_splits = tf.split(images, num_gpus, 0)
            labels_splits = [[] for _ in xrange(num_gpus)]
            for label in labels:
                label_splits = tf.split(label, num_gpus, 0)
                for i in xrange(num_gpus):
                    labels_splits[i].append(label_splits[i])

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
        variables_to_average = (slim.get_model_variables(scope=(FLAGS.basenet+'/Mixed')) 
                + slim.get_model_variables(scope='convbox_detection'))
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
        # start running. 
        sess = tf.Session(config=tf.ConfigProto(
                intra_op_parallelism_threads=4,
                allow_soft_placement=True,
                log_device_placement=False,
                gpu_options=gpu_options))

        sess.run(init)
        # load pretrained model or resume model.
        model_init(sess)

        # start the tensorflow and custom QueueRunner's thraeds.
        # tf.train.start_queue_runners(sess=sess)
        custom_runner.start_threads(sess)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        while True:
            start_time = time.time()
            _, loss_value, lr_value, step = sess.run([train_op, loss, lr, global_step])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            
            if step % 10 == 0:
                examples_per_sec = FLAGS.batch_size / float(duration)
                format_str = ('%s: step %d, lr = %.5f, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print format_str % (datetime.now(), step, lr_value, loss_value,
                                    examples_per_sec, duration)

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 2000 == 0 or step == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                if step == FLAGS.max_steps:
                    break


def main(unused_argv=None):
    FLAGS(sys.argv)
    if FLAGS.mode == 'train':
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
    tf.app.run()
