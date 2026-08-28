import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

import constants
from datasets import ThermalSequenceDataset
import models

if __name__ == '__main__':

    input_length = 64
    output_length = 16
    step = 64

    hidden_dim = 128
    num_layers = 1

    input_dir = Path(r"D:\MASON\Data\LSTM\test_in")
    output_dir = Path(r"D:\MASON\Data\LSTM\test_out")

    model = models.ThermalLSTM(hidden_dim=hidden_dim,
                            input_sequence_length=input_length,
                            output_sequence_length=output_length,
                            num_layers=num_layers)

    if (output_dir / 'checkpoint.pt').is_file():
        model.load_state_dict(torch.load(output_dir / 'checkpoint.pt', weights_only=True))

    model.to(constants.TORCH_DEVICE)

    data = ThermalSequenceDataset(Path(r"D:/MASON/Data/LSTM/test_in"), input_seq_len= 64, output_seq_len = 16, step = 64)

    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)

    loader = iter(loader)

    fix, ax = plt.subplots(3, 3, layout='constrained')

    model.eval()
    with torch.no_grad():
        for i in range(3):
            for j in range(3):

                inpt, target = next(loader)
                length = len(inpt.numpy().squeeze()) + len(target.numpy().squeeze())
                switch_pt = len(inpt.numpy().squeeze()) - 1

                inpt = inpt.to(constants.TORCH_DEVICE)
                target = target.to(constants.TORCH_DEVICE)

                result = model(inpt)

                result = result.cpu().numpy().squeeze()
                inpt = inpt.cpu().numpy().squeeze()
                target = target.cpu().numpy().squeeze()
                
                ax[i][j].plot(list(range(length)), np.append(inpt * (data.norm_max - data.norm_min) - data.norm_min, target * (data.norm_max - data.norm_min) - data.norm_min), label='True Curve')
                ax[i][j].plot(list(range(length)), np.append(inpt * (data.norm_max - data.norm_min) - data.norm_min, result * (data.norm_max - data.norm_min) - data.norm_min), label='Predicted Curve')
                ax[i][j].scatter(switch_pt, inpt[-1] * (data.norm_max - data.norm_min) - data.norm_min, color='red')

        ax[0][0].legend()

    plt.show()