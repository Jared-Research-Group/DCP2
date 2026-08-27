import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

import torch_testing
from torch_testing import ThermalLSTM

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(Path(r"D:/MASON/Data/LSTM/weights.pt"), weights_only=False)

    data = torch_testing.ThermalSequenceDataset([Path(r"D:\MASON\Data\LSTM\sequences.npy")], input_seq_len= 64, output_seq_len = 16, step = 10)

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

                inpt = inpt.to(device)
                target = target.to(device)

                result = model(inpt)

                result = result.cpu().numpy().squeeze()
                inpt = inpt.cpu().numpy().squeeze()
                target = target.cpu().numpy().squeeze()
                
                ax[i][j].plot(list(range(length)), np.append(inpt, target), label='True Curve')
                ax[i][j].plot(list(range(length)), np.append(inpt, result), label='Predicted Curve')
                ax[i][j].scatter(switch_pt, inpt[-1], color='red')

        ax[0][0].legend()

    plt.show()