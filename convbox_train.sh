#!/bin/bash

dataset="pascal_voc"
train_mode="serial"
log_file="backup/log/coco-dilation-aug-ems-final-512-3.log"

flagfile=flags/${dataset}_train_flags

# Run the fine-tuning on the detection data set  
# starting from the pre-trained Imagenet-v2 model.
case ${train_mode} in
    "parallel") python convbox/convbox_train.py \
                    --flagfile=${flagfile} | tee -a ${log_file}
    ;;
    "serial") python convbox/convbox_serial_train.py \
                    --flagfile=${flagfile} | tee -a ${log_file}
    ;;
    *) echo "error."
    ;;
esac
