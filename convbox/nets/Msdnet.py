from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math
slim = tf.contrib.slim
class param(object):
    nScales = 3
    reduction =0.5
    base = 4
    nBlocks = 10
    stepmode = 'even'
    step = 2
    bottleneck = True
    grFactor = [1,2,4,4]
    prune = 'max'
    bnFactor = [1,2,4,4]
    initChannels=32
    growthRate=6
    
def build(input, nChannels, nOutChannels, type, bottleneck, bnWidth):
    innerChannels = nChannels
    if not bnWidth :
      bnWidth = 4
    with tf.variable_scope(type):
        if bottleneck:
          innerChannels = min(innerChannels, bnWidth * nOutChannels)
          input = slim.conv2d(input, innerChannels, 1, stride=1, padding='VALID',normalizer_fn =slim.batch_norm, scope='conv_bottleneck');
        if type == 'normal':
          output = slim.conv2d(input, nOutChannels, 3,stride=1,padding='SAME',normalizer_fn =slim.batch_norm,scope = 'conv_normal')
        elif type == 'down':
          output = slim.conv2d(input, nOutChannels, 3,stride=2,padding='SAME',normalizer_fn =slim.batch_norm,scope = 'conv_dowm')
        elif type == 'up':
          output = slim.conv2d_transpose(input, nOutChannels, 3,stride=2,padding='SAME',normalizer_fn =slim.batch_norm,scope = 'conv_up')
    
    return output
    
def build_net_normal(input, nChannels, nOutChannels, bottleneck, bnWidth):
    #print (input)
    output1 = build(input[0], nChannels, nOutChannels, 'normal', bottleneck, bnWidth)
    output = tf.concat([input[0], output1], 3)
    return output
    
  
def build_net_down_normal(input, nChannels1, nChannels2, nOutChannels, bottleneck, bnWidth1, bnWidth2, scale,isTrans):
    assert nOutChannels % 2 == 0, 'Growth rate invalid!'
    if isTrans:
      output1 = build(input[scale], nChannels1, nOutChannels/2, 'down', bottleneck, bnWidth1)
      output2 = build(input[scale+1], nChannels2, nOutChannels/2, 'normal', bottleneck, bnWidth2)
      output = tf.concat([input[scale+1], output1, output2], 3)
    else:
      output1 = build(input[scale-1], nChannels1, nOutChannels/2, 'down', bottleneck, bnWidth1)
      output2 = build(input[scale], nChannels2, nOutChannels/2, 'normal', bottleneck, bnWidth2)
      output = tf.concat([input[scale], output1, output2], 3)
    return output
    
def MSDNet_Layer_first(input, Cins, Couts, args):
    '''
     input : a tensor (orginal image)
     output: a table of nScale tensors
     
    '''
    output=[]
    #print (args.nScales)
    
    for i in range(args.nScales):
        with tf.variable_scope("scale%d" %i):
          if(i==0):
            output_s = slim.conv2d(input, Couts*args.grFactor[0], 3, stride=1, padding='SAME', normalizer_fn =slim.batch_norm, scope='conv1')
            
          else:
            output_s = slim.conv2d(output_s, Couts*args.grFactor[i], 3, stride=2, padding='SAME', normalizer_fn =slim.batch_norm, scope='conv1')
          output.append(output_s)
    return output
    
def MSDNet_Layer(input, nIn, nOutc, args, inScales, outScales):
    '''
     input: a table of `nScales` tensors
     output: a table of `nScales` tensors
    '''
    #print (input)
    outputs=[]
    discard = inScales - outScales
    assert discard<=1, 'Double check inScales {0} and outScales {1}'.format(inScales,outScales)
    offset = args.nScales - outScales
    isTrans = outScales<inScales  
    #print ("tran:",isTrans,'outscale:',outScales)
    for i in range(outScales):
        with tf.variable_scope("scale%d" %i):
            if i==0:
                if isTrans:
                    nIn1, nIn2, nOut = nIn*args.grFactor[offset-1], nIn*args.grFactor[offset+1-1], args.grFactor[offset+1-1]*nOutc
                    output = build_net_down_normal(input, nIn1, nIn2, nOut, args.bottleneck, args.bnFactor[offset-1], args.bnFactor[offset+1-1],i,isTrans)
                    
                else :
                    output = build_net_normal(input, nIn*args.grFactor[offset+1-1], args.grFactor[offset+1-1]*nOutc, args.bottleneck, args.bnFactor[offset+1-1])
                outputs.append(output)
            else :
                nIn1, nIn2, nOut = nIn*args.grFactor[offset+i-1], nIn*args.grFactor[offset+i], args.grFactor[offset+i]*nOutc
                output = build_net_down_normal(input,nIn1, nIn2, nOut, args.bottleneck, args.bnFactor[offset+i-1], args.bnFactor[offset+i],i,isTrans)
                outputs.append(output)
    
    #print (outputs)
    
    return outputs
    
def build_transition(input, nIn, nOut, outScales, offset, args):
    output=[]
    with tf.variable_scope('transition'):
      for i in range(outScales):
        with tf.variable_scope('scale%d' %i):
          output_s = slim.conv2d(input[i], nOut * args.grFactor[offset + i], 1, stride=1, padding='VALID', scope='conv1')
          output.append(output_s)
    return output
         
def build_block(input, inChannels, args, step, layer_all, layer_curr,blockname):
    nIn = inChannels
    with tf.variable_scope(blockname):
        if layer_curr==0:
          input = MSDNet_Layer_first(input,3,inChannels,args)
          #print ("first layer feature size",input)
        
        
        for i in range(step):
          inScales, outScales = args.nScales, args.nScales
          layer_curr = layer_curr+1
          #add inscale outscale computation here
          if args.prune == 'min':
            inScales = min(args.nScales, layer_all - layer_curr + 2)
            outScales = min(args.nScales, layer_all - layer_curr + 1)
          elif args.prune=='max':
            interval = math.ceil(layer_all/args.nScales)
            #print (layer_all,args.nScales)
            #print ('interval',interval)
            inScales = args.nScales - math.floor((max(0, layer_curr -2))/interval)
            #print ('inScales',inScales)
            outScales = args.nScales - math.floor((layer_curr -1)/interval)
            #print ('outScales',outScales)
          inScales = int(inScales)
          outScales = int(outScales)
          #print('|', 'inScales ', inScales, 'outScales ', outScales , '|') 
          with tf.variable_scope('step_%d' %i):
            #print (len(input))
            input = MSDNet_Layer(input,nIn,args.growthRate,args,inScales,outScales)
            #print (blockname,"shape",input)
          nIn = nIn + args.growthRate
          if args.prune == 'max' and inScales > outScales and args.reduction > 0 :
            offset = args.nScales - outScales
            input = build_transition(input, nIn, math.floor(args.reduction*nIn), outScales, offset, args)
            nIn = math.floor(args.reduction*nIn)
            print('|', 'Transition layer inserted!', '\t\t|')
          elif args.prune == 'min' and args.reduction >0 and ((layer_curr == math.floor(layer_all/3) or layer_curr == math.floor(2*layer_all/3))):
            offset = args.nScales - outScales
            input = build_transition(input, nIn, math.floor(args.reduction*nIn), outScales, offset, args)
            nIn = math.floor(args.reduction*nIn)
            print('|', 'Transition layer inserted!', '\t\t|')
    #print (type(input))
    print ('------')
    return input, nIn

def build_detector_pascal(input, inChannels, scopename,args):
    #input [batch,112,112,inChannels]
    #interChannels1, interChannels2 = 128, 128
    with tf.variable_scope(scopename):
      #with slim.arg_scope([slim.conv2d],normalizer_fn=None, normalizer_params=None):
        output = slim.conv2d(input, inChannels, 3, stride=1,padding='SAME', scope='conv1')
        output = slim.conv2d(output, inChannels, 3, stride=2,padding='SAME', scope='conv2')
        output = slim.conv2d(output, inChannels, 3, stride=1,padding='SAME', scope='conv3')
        #output = slim.conv2d(output, inChannels, 3, stride=1,padding='SAME', scope='conv3',activation_fn=None)
        #dets = slim.conv2d(output, final_channels, 1, stride=1,padding='SAME', scope='conv4')
    return output#[batch,14,14,final_channels]
    

 
def Msdnet_base(inputs,scope=None):
    args = param()
    end_points = {}
    nChannels = 32
    nblocks = 10
    nIn = nChannels
    layer_curr = 0
    layer_all = 4
    steps=[0]*nblocks
    steps[0]=4
    outputs=[]
    for i in range(1,nblocks):
     if True:
       steps[i]=2
     else :
       pass
     layer_all= layer_all+steps[i]
    print("building network of steps: ")
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    }
    with tf.variable_scope(scope,'Msdnet',[inputs]):
      #with slim.arg_scope([slim.conv2d, slim.fully_connected],
      #                  weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
       #                 weights_regularizer=slim.l2_regularizer(1e-4),
       #                 normalizer_fn=slim.batch_norm,
       #                 normalizer_params=batch_norm_params):
        #with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=True):
            inputs = slim.conv2d(inputs, nChannels, 7, stride=2,padding='SAME', scope='conv_7x7')
            inputs = slim.max_pool2d(inputs,2,stride=2,padding = 'SAME',scope='max_pooling')
            print (inputs)
            for i in range(nblocks):
                blockname = 'b_'+str(i)
                if layer_curr==0:
                    net,nIn = build_block(inputs,nIn,args,steps[i],layer_all,layer_curr,blockname)
                else:
                    net,nIn = build_block(net,nIn,args,steps[i],layer_all,layer_curr,blockname)
                layer_curr = layer_curr + steps[i]
            detectname='detector_'+str(10)
            dets = build_detector_pascal(net[-1],nIn*args.grFactor[args.nScales] , detectname,args)
                #outputs.append(dets)
            
    print (outputs)
    return dets,end_points
        
    