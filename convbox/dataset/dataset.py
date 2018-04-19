# --------------------------------------------------------
# ConvBox
# Copyright (c) 2016 HUST
# Licensed under The MIT License [see LICENSE for details]
# Written by Ckb
# --------------------------------------------------------

import lmdb
import random
from random import shuffle
from datetime import datetime


class dataset(object):
    """
    Image dataset.
    """

    def __init__(self, params):
        self._name = params['name']
        self._lmdb_file = params['lmdb_file']
        self.init_lmdb
        print '{}: Dataset initialized with {} images.'.format( 
                datetime.now(), self.num_images)

    @property
    def init_lmdb(self):
        print '{}: Load lmdb data from {}.'.format(
                datetime.now(), self._lmdb_file)
        self._env = lmdb.open(self._lmdb_file, readonly=True)
        print '{}: Load lmdb successfully.'.format(
                datetime.now())
        with self._env.begin() as txn:
            self._lmdb_keys = [k for k, _ in txn.cursor()]
        self._env.close()
        self._env = lmdb.open(self._lmdb_file, readonly=True)
        self._cur = 0

    @property
    def name(self):
        return self._name

    @property
    def num_images(self):
        return len(self._lmdb_keys)

    @property
    def next_raw_datum(self):
        if self._cur == self.num_images:
            self._cur = 0
            shuffle(self._lmdb_keys)
        lmdb_key = self._lmdb_keys[self._cur]
        
        if self._cur % 10000 == 0:
            self._env.close()
            self._env = lmdb.open(self._lmdb_file, readonly=True)
        with self._env.begin() as txn:
            raw_datum = txn.get(lmdb_key)
        self._cur += 1
        return raw_datum
