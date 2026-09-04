from pathlib import Path
import numpy as np
#import pysr
import os
import sympy as sp
import zarr
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import helper_functions
import constants
import datasets
import models
import nn_framework

args = {
    'num_epochs': int(3e6),
    'report_step': int(1e3),
    'lr': .5,
    'lr_scheduler_args': {'patience': int(1e4)},
    'loader_args': {'batch_size': 2**12},
}

#input_dir = Path(r"D:\MASON\Data\LSTM\in")
#output_dir = Path(r"D:\MASON\Data\LSTM\out")

input_dir = Path(r"D:\MASON\Data\LSTM\test_in")
output_dir = Path(r"D:\MASON\Data\LSTM\test_good_result")


class ThermalHistoryPredictor(nn_framework.NeuralNet):

    @nn_framework.NeuralNet.initializer
    def __init__(self, raw_data_path: Path, output_directory: Path, **kwargs):

        self._handle_general_constants(raw_data_path, output_directory, **kwargs)

        self.input_seq_len   = 64
        self.output_seq_len  = 16
        self.seq_step        = 32

        self.hidden_dim      = 128
        self.num_lstm_layers = 1

        self.model_type = models.ThermalLSTM

        self.model_args = {
            'hidden_dim': self.hidden_dim,
            'input_sequence_length': self.input_seq_len,
            'output_sequence_length': self.output_seq_len,
            'num_layers': self.num_lstm_layers
        }

        return

    def _preprocess_data(self):

        input_files = [file for file in self.input_path.iterdir() if file.suffix == '.npy']
        example_data = np.load(input_files[0], allow_pickle=True)

        file_len = (example_data.shape[0] - (self.input_seq_len + self.output_seq_len)) // self.seq_step + 1

        self.data = zarr.create_group(store = self.output_dir / 'data.zarr')

        dataset = self.data.create_group('dataset')

        #input_chunk_size = int((1024 * 1024) / (self.input_seq_len * 8))
        input_chunk_size = 1
        input_shard_size = input_chunk_size * 1024
        input = dataset.create_array('input',
                               shape = (file_len * len(input_files), self.input_seq_len),
                               chunks = (input_chunk_size, self.input_seq_len),
                               shards = (input_shard_size, self.input_seq_len),
                               dtype = 'float64'
                               )

        #output_chunk_size = int((1024 * 1024) / (self.output_seq_len * 8))
        output_chunk_size = 1
        output_shard_size = output_chunk_size * 1024
        target = dataset.create_array('target',
                             shape = (file_len * len(input_files), self.output_seq_len),
                             chunks = (output_chunk_size, self.output_seq_len),
                             shards = (output_shard_size, self.output_seq_len),
                             dtype = 'float64')

        input_sequences = []
        target_sequences = []
        zarr_input_idx = 0
        zarr_output_idx = 0

        for file in tqdm(input_files, ascii = True):
            data = np.load(file, allow_pickle = True)

            # build sequences of desired length
            for idx in np.arange(0, data.shape[0] - (self.input_seq_len + self.output_seq_len), self.seq_step):
                input_idx  = idx
                target_idx = input_idx + self.input_seq_len

                input_sequences.append(data[input_idx :input_idx  + self.input_seq_len])
                target_sequences.append(data[target_idx:target_idx + self.output_seq_len])

                # write in chunk-sized batches
                if len(input_sequences) == input_shard_size:
                    self._safe_zarr_write(input, zarr_input_idx, input_sequences)
                    input_sequences = []
                    zarr_input_idx += input_shard_size

                if len(target_sequences) == output_shard_size:
                    self._safe_zarr_write(target, zarr_output_idx, target_sequences)
                    target_sequences = []
                    zarr_output_idx += output_shard_size

        # catch remaining semi-chunks
        self._safe_zarr_write(input, zarr_input_idx, input_sequences)
        self._safe_zarr_write(target, zarr_output_idx, target_sequences)

        return

    @staticmethod
    def _safe_zarr_write(array, idx, data, attempts=10, waittime=1e-3, dtype=np.float64):

        for attempt in range(attempts):
            try:
                array[idx:idx + len(data), :] = np.array(data, dtype=dtype)
                return
            except PermissionError as e:
                if attempt == attempts - 1:
                    raise e
                time.sleep(waittime)

    def test(self):
        super(ThermalHistoryPredictor, self).test()

        loader_args = {
            'batch_size':         1,
            'shuffle':            True,
            'num_workers':        0,
        }

        plot_loader = iter(torch.utils.data.DataLoader(self.training_dataset, **loader_args))

        fix, ax = plt.subplots(3, 3, layout='constrained')

        with torch.no_grad():
            for i in range(3):
                for j in range(3):

                    inpt, target = next(plot_loader)
                    length = len(inpt.cpu().numpy().squeeze()) + len(target.cpu().numpy().squeeze())
                    switch_pt = len(inpt.cpu().numpy().squeeze()) - 1

                    inpt = inpt.to(constants.TORCH_DEVICE)
                    target = target.to(constants.TORCH_DEVICE)

                    result = self.model(inpt)

                    result = result.cpu().numpy().squeeze()
                    inpt = inpt.cpu().numpy().squeeze()
                    target = target.cpu().numpy().squeeze()
                    
                    ax[i][j].plot(list(range(length)), self.reverse_global_normalization(np.append(inpt, target)), label='True Curve')
                    ax[i][j].plot(list(range(length)), self.reverse_global_normalization(np.append(inpt, result)), label='Predicted Curve')
                    ax[i][j].scatter(switch_pt, self.reverse_global_normalization(inpt[-1]), color='red')

            ax[0][0].legend()

        plt.show()

    def apply_global_normalization(self):

        # use FLIR calibration to get max possible temperature
        #high_fit = pysr.PySRRegressor().from_file(run_directory=os.getcwd() + '/FLIR_fits/High', model_selection='best', verbosity=0)

        #x = sp.symbols('FLIR_Intensity')
        #fn = sp.lambdify(x, high_fit.sympy(11), modules='numpy')

        #self.norm_max = fn(2**16 - 1) - 273.15
        self.norm_max = 1150
        self.norm_min = 0 # assume minimum temperature of 0 C

        assert isinstance(self.data['dataset/input'],  zarr.Array)
        assert isinstance(self.data['dataset/target'], zarr.Array)

        self.data['dataset/input']  = (self.data['dataset/input'][...]  - self.norm_min) / (self.norm_max - self.norm_min)
        self.data['dataset/target'] = (self.data['dataset/target'][...] - self.norm_min) / (self.norm_max - self.norm_min)

        return

    def reverse_global_normalization(self, data):

        # use FLIR calibration to get max possible temperature
        #high_fit = pysr.PySRRegressor().from_file(run_directory=os.getcwd() + '/FLIR_fits/High', model_selection='best', verbosity=0)

        #x = sp.symbols('FLIR_Intensity')
        #fn = sp.lambdify(x, high_fit.sympy(11), modules='numpy')

        #self.norm_max = fn(2**16 - 1) - 273.15
        self.norm_max = 1150
        self.norm_min = 0 # assume minimum temperature of 0 C

        return data * (self.norm_max - self.norm_min) + self.norm_min

# ========================================================================================

if __name__ == '__main__':

    NN = ThermalHistoryPredictor(input_dir, output_dir, **args)

    #model = NN.train()

    NN.test()