import os
import h5py
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

class Dataset(object):
    def __init__(self, data, stats):
        self.__data = data
        self.mean = stats['mean']
        self.std = stats['std']

    def get_data(self, type):
        return self.__data[type]

    def get_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def get_len(self, type):
        return len(self.__data[type])

    def z_inverse(self, type):
        return self.__data[type] * self.std + self.mean

def seq_gen(len_seq, data_seq, offset, n_frame, n_route, n_modal, day_slot, C_0=1):
    n_slot = day_slot

    tmp_seq = np.zeros((len_seq * n_slot, n_frame, n_route,  n_modal, C_0))
    for i in range(len_seq):
        for j in range(n_slot):
            end = (i + offset) * day_slot + j + 1
            sta = end - n_frame
            if sta >= 0:
                tmp_seq[i * n_slot + j, :, :, :, :] = np.reshape(data_seq[sta:end, :, :], [n_frame, n_route, n_modal, C_0])
    return tmp_seq

def data_gen(file_path, data_config, n_route, n_frame=21, n_modal=4, day_slot=288):
    n_train, n_val, n_test = data_config
    # generate training, validation and test data
    try:
        h = h5py.File(file_path)
        data_seq = h["data"][:]
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')
    print("DATA SIZE: ", data_seq.shape)
    seq_train = seq_gen(n_train, data_seq, 0, n_frame, n_route, n_modal, day_slot)
    seq_train = seq_train[n_frame:]
    seq_val = seq_gen(n_val, data_seq, n_train, n_frame, n_route,  n_modal, day_slot)
    seq_test = seq_gen(n_test, data_seq, n_train + n_val, n_frame, n_route,  n_modal, day_slot)
    # x_stats: dict, the stats for the train dataset, including the value of mean and standard deviation.      
    x_stats = {'mean': np.mean(seq_train), 'std': np.std(seq_train)}
    x_train = z_score(seq_train, x_stats['mean'], x_stats['std'])
    x_val = z_score(seq_val, x_stats['mean'], x_stats['std'])
    x_test = z_score(seq_test, x_stats['mean'], x_stats['std'])        
    x_data = {'train': x_train, 'val': x_val, 'test': x_test}
    dataset = Dataset(x_data, x_stats)
    return dataset


def gen_batch(inputs, batch_size, dynamic_batch=False, shuffle=False, period=None):
    len_inputs = len(inputs)
    if shuffle:
        idx = np.arange(len_inputs)
        np.random.shuffle(idx)
    for start_idx in range(0, len_inputs, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len_inputs:
            if dynamic_batch:
                end_idx = len_inputs
            else:
                break
        if shuffle:
            slide = idx[start_idx:end_idx]
        else:
            slide = slice(start_idx, end_idx)
        yield inputs[slide]
        
def z_score(x, mean, std):
    return (x - mean) / std

def z_inverse(x, mean, std):
    return x * std + mean

def RMSE(v, v_):
    return np.mean((v_ - v) ** 2, axis=(0, 2, 4)) ** 0.5

def MAE(v, v_):    
    return np.mean(np.abs(v_ - v), axis=(0, 2, 4))                                                                                                                                       

def get_metric(y, y_, x_stats):
    y = z_inverse(y, x_stats['mean'], x_stats['std'])
    y_ = z_inverse(y_, x_stats['mean'], x_stats['std'])
    # rmse, mae : (horizon, sources+1)
    rmse = RMSE(y, y_)
    mae = MAE(y, y_)
    return mae, rmse
