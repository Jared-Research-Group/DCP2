import numpy as np
from pathlib import Path

import torch
import json


class ThermalSequenceDataset(torch.utils.data.Dataset):

    def __init__(self, path: Path, input_seq_len: int, output_seq_len: int, step: int = 30):

        self.path = path
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.step = step

        with open(self.path / 'data.json', 'r') as file:
            self.time_len = json.load(file)['time_length']

        self.files = [file for file in self.path.iterdir() if file.suffix != '.json']

        """
        self.data = None

        seq_len = self.input_seq_len + output_seq_len

        for path in self.paths:
            long_sequences = np.load(path, allow_pickle = True)
            long_sequences = long_sequences.reshape((np.prod(long_sequences.shape[:2]),) + long_sequences.shape[2:])
            long_sequences = np.astype(long_sequences, np.float32)

            data = np.empty((np.prod(long_sequences.shape[:-1]), int((long_sequences.shape[-1] - seq_len) / step) + 1, seq_len), dtype=np.float32)

            for i, idx in enumerate(range(0, long_sequences.shape[-1] - seq_len, step)):
                data[:, i, :] = long_sequences[:, idx:(idx + seq_len)]

            data = data.reshape(-1, seq_len)

            if self.data is None:
                self.data = data
            else:
                self.data = np.append(self.data, data, axis=0)

            print(f'finished building dataset.\nshape: {self.data.shape}\n')
            """
    def __len__(self):

        return len(self.files) * (self.time_len - (self.input_seq_len + self.output_seq_len) // self.step)

    def __getitem__(self, idx):

        file_idx  = idx // (self.time_len - (self.input_seq_len + self.output_seq_len) // self.step)
        start_idx = idx %  (self.time_len - (self.input_seq_len + self.output_seq_len) // self.step)

        file = np.load(self.files[file_idx], allow_pickle=True)

        input  = torch.from_numpy(file[start_idx                      : start_idx + self.input_seq_len])
        target = torch.from_numpy(file[start_idx + self.input_seq_len : start_idx + self.input_seq_len + self.output_seq_len])
        return input, target