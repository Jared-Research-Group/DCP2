import numpy as np
import matplotlib.pyplot as plt

import helper_functions

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device detected: {device}\n')

class ThermalSequenceDataset(torch.utils.data.Dataset):

    def __init__(self, paths, input_seq_len, output_seq_len, step = 30):

        self.paths = paths
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.step = step

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

            print('finished building dataset...\n')

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        input  = torch.from_numpy(self.data[idx, :self.input_seq_len]).to(device)
        output = torch.from_numpy(self.data[idx, self.input_seq_len:]).to(device)
        return input, output


class ThermalLSTM(nn.Module):

    def __init__(self, hidden_dim, input_sequence_length, output_sequence_length, input_size = 1, batch_size = 1, num_layers = 1):

        super(ThermalLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.input_size = input_size

        self.input_sequence_length  = input_sequence_length
        self.output_sequence_length = output_sequence_length

        self.lstm = nn.LSTM(input_size,
                              hidden_dim,
                              num_layers = num_layers,
                              batch_first=True,
                              dtype = torch.float32)

        self.head = nn.Linear(hidden_dim, output_sequence_length)

    def forward(self, input):
        # input needs shape (batch_size, input_sequence_length, input_size)

        lstm_out, (h_n, c_n) = self.lstm(input.view(self.batch_size, self.input_sequence_length, self.input_size))
        output = self.head(h_n[-1])

        return output


if __name__ == '__main__':

    #paths = [helper_functions.selectFile()]
    paths = [r"E:\410SS DATA\modified datasets\L wall 400C Interpass 121225\data_collection_20251212_141949\temp_data_data_collection_20251212_141949\sequences.npy"]

    loader = torch.utils.data.DataLoader(ThermalSequenceDataset(paths, input_seq_len = 64, output_seq_len = 16, step = 60), batch_size=32, shuffle=True, drop_last=True)#, num_workers=4, pin_memory=True, )

    model = ThermalLSTM(hidden_dim = 32, input_sequence_length = 64, output_sequence_length = 16, batch_size=32).to(device)

    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=.00005)

    """
    model.eval()
    with torch.no_grad():

        inputs, targets  = next(iter(loader))
        outputs = model(inputs)

        losses = loss_function(outputs, targets)
        print(f'Average MSE: {losses.mean()}')
    """

    loss_hist = []
    model.train()
    for epoch in range(300):

        it = iter(loader)
        losses = []
        for ins, target in it:

            model.zero_grad()

            result = model(ins)
            loss = loss_function(result, target)

            loss.backward()
            optimizer.step()

            losses.append(float(loss.mean().detach()))

        loss_hist.append(np.mean(losses))
        print(f'epoch: {epoch}\tloss: {loss_hist[-1]:>8.2f}', end='\r', flush=True)


    print()
    plt.semilogy(np.arange(len(loss_hist)), loss_hist)
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.title('LSTM Loss History')
    plt.savefig('hist.png')
    plt.show()
    
