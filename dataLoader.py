from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from scipy import signal
import pickle
import os
import lmdb
import pickle

class My_Dataset(Dataset):

    def __init__(self, path, label):
        self.path = path
        self.label = label
        self.lmdb_env = None
        self.txn = None
        self.length = 0


    def _init_db(self):
        self.lmdb_env = lmdb.open(self.path, subdir=os.path.isdir(self.path),
            readonly=True, lock=False,
            readahead=False, meminit=False)
        self.txn = self.lmdb_env.begin()

    def get_key_name(self, label, index):
        if label == 'train':
            name_x = f'train_data_{index}'
            name_y = f'train_label_{index}'

        elif label == 'test':
            name_x = f'test_data_{index}'
            name_y = f'test_label_{index}'

        elif label =='val':
            name_x = f'val_data_{index}'
            name_y = f'val_label_{index}'
        return  name_x, name_y
    
    def __len__(self):
        if self.lmdb_env is None:
            self._init_db()
        if self.label == 'train':
            key = 'train_len'
        elif self.label == 'test':
            key = 'test_len'
        elif self.label =='val':
            key = 'val_len'
        self.length = pickle.loads(self.txn.get(key.encode('utf-8')))
        return self.length
    
    def read_lmdb(self, index):
        name_x, name_y = self.get_key_name(self.label, index)
        # with lmdb_env.begin() as lmdb_txn:
        x = pickle.loads(self.txn.get(name_x.encode('utf-8')))
        y = pickle.loads(self.txn.get(name_y.encode('utf-8')))
        return x, y

    def __getitem__(self, item):
        if self.lmdb_env is None:
            self._init_db()
        X_sample, y_sample = self.read_lmdb(item)
        label = np.delete(y_sample,(9,11,16), axis=1)
        return X_sample, label
    
    def close(self):
        self.lmdb_env.close()