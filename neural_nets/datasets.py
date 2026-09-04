import numpy as np
import zarr
import torch

import constants

# TODO: handle grouping
class zarr_generic(torch.utils.data.Dataset):

    def __init__(self, zarr_group: zarr.Group):
        self.zarr_group = zarr_group

        self.input  = self.zarr_group['input']
        self.target = self.zarr_group['target']

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):

        input  = torch.from_numpy(self.input[idx])
        target = torch.from_numpy(self.target[idx])

        return input, target

class in_memory_generic(torch.utils.data.Dataset):

    def __init__(self, zarr_group: zarr.Group):

        self.input = torch.from_numpy(zarr_group['input'][:]).to(constants.TORCH_DEVICE)
        self.target = torch.from_numpy(zarr_group['target'][:]).to(constants.TORCH_DEVICE)

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx) -> tuple[torch.tensor, torch.tensor]:

        input  = self.input[idx]
        target = self.target[idx]

        return input, target