import numpy as np
import zarr
from pathlib import Path

import torch
import json

# TODO: handle grouping
class zarr_generic(torch.utils.data.Dataset):

    def __init__(self, zarr_group):
        self.zarr_group = zarr_group

        self.input  = self.zarr_group['input']
        self.target = self.zarr_group['target']

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):

        input  = torch.from_numpy(self.input[idx])
        target = torch.from_numpy(self.target[idx])

"""
class ThermalSequenceDataset(torch.utils.data.Dataset):

    def __init__(self, path: Path, input_seq_len: int, output_seq_len: int, step: int = 30):

        self.path = path
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.step = step

        self.norm_max = 1150
        self.norm_min = 0

        with open(self.path / 'data.json', 'r') as file:
            self.time_len = json.load(file)['time_length']

        self.files = [file for file in self.path.iterdir() if file.suffix != '.json']

        self.num_segments = (self.time_len - (self.input_seq_len + self.output_seq_len)) // self.step

    def __len__(self):

        return len(self.files) * self.num_segments

    def __getitem__(self, idx):

        file_idx  = idx // self.num_segments
        start_idx = (idx %  self.num_segments) * self.step

        file = np.load(self.files[file_idx], allow_pickle=True)
        file = file.astype(np.float32)

        input  = torch.from_numpy(file[start_idx                      : start_idx + self.input_seq_len])
        target = torch.from_numpy(file[start_idx + self.input_seq_len : start_idx + self.input_seq_len + self.output_seq_len])
        return (input - self.norm_min) / (self.norm_max - self.norm_min), (target - self.norm_min) / (self.norm_max - self.norm_min)
"""