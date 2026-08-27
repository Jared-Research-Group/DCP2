from typing import Final

import torch
TORCH_DEVICE: Final[torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu")