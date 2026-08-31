from pathlib import Path
import numpy as np
import pysr
import os
import sympy as sp
import zarr
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import helper_functions

import datasets
import models
import nn_framework

args = {
    'num_epochs': 10000,
    'report_step': 2
}

input_dir = Path(r"D:\MASON\Data\LSTM\in")
output_dir = Path(r"D:\MASON\Data\LSTM\out")

class ThermalHistoryPredictor(nn_framework.NeuralNet):

    def __init__(self, raw_data_path, output_directory, **kwargs):

        super(ThermalHistoryPredictor, self).__init__(raw_data_path, output_directory, **kwargs)

        self.model_type = models.ThermalLSTM

        self.input_seq_len   = 64
        self.output_seq_len  = 64
        self.seq_step        = 32

        self.hidden_dim      = 128
        self.num_lstm_layers = 1

        self.handle_kwargs(**kwargs)

        self.model_args = {
            'hidden_dim': self.hidden_dim,
            'input_sequence_length': self.input_seq_len,
            'output_sequence_length': self.output_seq_len,
            'num_layers': self.num_lstm_layers
        }

    def preprocess_data(self):

        self.data = zarr.create_group(store = self.output_dir / 'dataset')
        dataset = self.data.create_group('dataset')

        input = dataset.create_array('input',
                               shape = (0, self.input_seq_len),
                               chunks = (1, self.input_seq_len),
                               dtype = 'float64'
                               )

        target = dataset.create_array('target',
                             shape = (0, self.output_seq_len),
                             chunks = (1, self.output_seq_len),
                             dtype = 'float64')

        input_files = [file for file in self.input_path.iterdir() if file.suffix == 'npy']
        for file in tqdm(input_files, ascii = True):
            data = np.load(file, allow_pickle = True)

            for idx in range(0, len(data) - (self.input_seq_len + self.output_seq_len), self.seq_step):
                input_idx  = idx
                target_idx = input_idx + self.input_seq_len

                input .append(data[input_idx :input_idx  + self.input_seq_len])
                target.append(data[target_idx:target_idx + self.output_seq_len])

        return

    def apply_global_normalization(self):

        # use FLIR calibration to get max possible temperature
        high_fit = pysr.PySRRegressor().from_file(run_directory=os.getcwd() + '/FLIR_fits/High', model_selection='best', verbosity=0)

        x = sp.symbols('FLIR_Intensity')
        fn = sp.lambdify(x, high_fit.sympy(11), modules='numpy')

        self.norm_max = fn(2**16 - 1) - 273.15
        self.norm_min = 0 # assume minimum temperature of 0 C

        input  = self.data['dataset/input']
        target = self.data['dataset/target']

        assert input  is zarr.Array
        assert target is zarr.Array

        input  = (input  - self.norm_min) / (self.norm_max - self.norm_min)
        target = (target - self.norm_min) / (self.norm_max - self.norm_min)

        return

    def reverse_global_normalization(self, data):

        # use FLIR calibration to get max possible temperature
        high_fit = pysr.PySRRegressor().from_file(run_directory=os.getcwd() + '/FLIR_fits/High', model_selection='best', verbosity=0)

        x = sp.symbols('FLIR_Intensity')
        fn = sp.lambdify(x, high_fit.sympy(11), modules='numpy')

        self.norm_max = fn(2**16 - 1) - 273.15
        self.norm_min = 0 # assume minimum temperature of 0 C

        return data * (self.norm_max - self.norm_min) + self.norm_min

# ========================================================================================

if __name__ == '__main__':

    NN = ThermalHistoryPredictor(input_dir, output_dir, **args)

    model = NN.train()