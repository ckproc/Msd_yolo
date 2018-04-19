#!/bin/bash

# download inception-v2 model.
mkdir -p model/inception-v2
wget http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz
tar zxvf inception_v2_2016_08_28.tar.gz
mv inception_v2_2016_08_28.tar.gz model/
mv inception_v2.ckpt model/inception-v2/
