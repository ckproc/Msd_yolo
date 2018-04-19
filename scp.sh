#!/bin/bash

# scp -P 1080 ckb@115.156.164.243:/home/ckb/Desktop/convbox2/backup/coco-dilation-aug-ems-final-512-3/model.ckpt-580000.* backup/coco-dilation-aug-ems-final-512-3/

# scp -P 1080 result/data/result.json ckb@115.156.164.243:/home/ckb/Desktop/result-coco/res/result-0.json
scp -P 1081 result/data/result.json ckb@115.156.164.243:/disk2/ckb/convbox2/result-coco/res/result-0.json
