# --------------------------------------------------------
# ConvBox
# Copyright (c) 2016 HUST
# Licensed under The MIT License [see LICENSE for details]
# Written by Ckb
# --------------------------------------------------------

import inception_v2
import inception_v3
import inception_v4
import Msdnet
def basenet(images, net_name):
    print net_name
    if net_name == 'InceptionV2':
        return inception_v2.inception_v2_base(images, scope='InceptionV2')
    elif net_name == 'InceptionV3':
        return inception_v3.inception_v3_base(images, scope='InceptionV3')
    elif net_name == 'InceptionV4':
        return inception_v4.inception_v4_base(images, scope='InceptionV4')
    elif net_name == 'Msdnet':
        return Msdnet.Msdnet_base(images,scope='Msdnet')
    else:
        raise ValueError('Basenet [%s] was not recognized.', net_name)
        
def basenet_arg_scope(net_name):
    if net_name == 'InceptionV2' or net_name == 'Msdnet':
        return inception_v2.inception_v2_arg_scope()
    elif net_name == 'InceptionV3':
        return inception_v3.inception_v3_arg_scope() 
    elif net_name == 'InceptionV4':
        return inception_v4.inception_v4_arg_scope()         
    else:
        raise ValueError('Basenet [%s] was not recognized.', net_name)
        
def output2X(net_name):
    if net_name == 'InceptionV2':
        return ('Mixed_4e', 576, 1536, 1024)
    elif net_name == 'InceptionV3':
        return ('Mixed_6e', 768, 2048, 1024)
    elif net_name == 'InceptionV4':
        return ('Mixed_6h', 1024, 1536, 1024)
    else:
        raise ValueError('Basenet [%s] was not recognized.', net_name)    