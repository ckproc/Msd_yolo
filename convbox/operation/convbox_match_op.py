# --------------------------------------------------------
# ConvBox
# Copyright (c) 2016 HUST
# Licensed under The MIT License [see LICENSE for details]
# Written by Ckb
# --------------------------------------------------------

import os.path

import tensorflow as tf

_convbox_match_module = tf.load_op_library(
    os.path.join(tf.resource_loader.get_data_files_path(),
                 'convbox_match.so'))
convbox_match = _convbox_match_module.convbox_match
