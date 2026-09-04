from typing import Final

import torch

# define constants
TORCH_DEVICE: Final[torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TORCH_DEVICE_MEMORY: Final[int]   
TORCH_DEVICE_MEMORY, _ = torch.cuda.memory.mem_get_info(TORCH_DEVICE)