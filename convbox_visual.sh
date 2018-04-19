#!/bin/bash

dataset="coco"

flagfile=flags/${dataset}_visual_flags

# Evaluate the fine-tuned model.
python convbox/convbox_visual.py \
    --flagfile=${flagfile}


