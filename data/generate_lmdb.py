
import os
import sys
import cv2
import lmdb
import numpy as np
from random import shuffle
sys.path.append("../convbox/dataset")
from proto.convbox_pb2 import Datum


def main(data_root, txt_file, lmdb_file):
    pair_datas = [line.rstrip('\n') for line in open(txt_file)]
    shuffle(pair_datas)

    N, count = 1000, 0
    env = lmdb.open(lmdb_file, map_size=int(1e12))
    txn = env.begin(write=True)
    for pair_data in pair_datas:
        train_pair = pair_data.split()

        im_id = os.path.splitext(os.path.split(train_pair[0])[1])[0]
        im_id = im_id.split('_')[-1]

        has_gt = (len(train_pair) == 2)
        if has_gt:
            la_id = os.path.splitext(os.path.split(train_pair[1])[1])[0]
            la_id = la_id.split('_')[-1]
            assert im_id == la_id

        im = cv2.imread(os.path.join(data_root, train_pair[0]))
        im = im[:, :, ::-1]   # BGR --> RGB

        datum = Datum()
        datum.id = im_id
        datum.height = im.shape[0]
        datum.width = im.shape[1]
        datum.channels = im.shape[2]
        datum.data = im.tobytes()
       
        if has_gt:
            # read boxes: [mid_x, mid_y, w, h]
            for line in open(os.path.join(data_root, train_pair[1]), 'r'):
                vline = line.strip().split()
                box = datum.box.add()
                box.id = int(vline[0])
                box.x = float(vline[1])
                box.y = float(vline[2])
                box.w = float(vline[3])
                box.h = float(vline[4])

        key_str = '{:0>8d}'.format(count)
        txn.put(key_str, datum.SerializeToString())
        
        count += 1
        if count % N == 0:
            txn.commit()
            txn = env.begin(write=True)
            print 'Deal with {} examples.'.format(count)

    if count % N != 0:
        txn.commit()
        print 'Deal with {} examples.'.format(count)


if __name__ == '__main__':
    data_root = sys.argv[1]
    txt_file = sys.argv[2]
    lmdb_file = sys.argv[3]
    main(data_root, txt_file, lmdb_file)
