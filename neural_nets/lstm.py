from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

import helper_functions

import datasets
import models
import nn_framework

input_length = 64
output_length = 16
step = 64

hidden_dim = 128
num_layers = 1

input_dir = Path(r"D:\MASON\Data\LSTM\in")
output_dir = Path(r"D:\MASON\Data\LSTM\out")

if __name__ == '__main__':

    #data_dir = Path(helper_functions.selectFolder())
    data_dir = input_dir

    data = datasets.ThermalSequenceDataset(data_dir,
                                           input_seq_len=input_length,
                                           output_seq_len=output_length, 
                                           step=step)

    train_data, vali_data, test_data = nn_framework.split_data(data)

    model = models.ThermalLSTM(hidden_dim=hidden_dim,
                            input_sequence_length=input_length,
                            output_sequence_length=output_length,
                            num_layers=num_layers)

    if (output_dir / 'checkpoint.pt').is_file():
        model.load_state_dict(torch.load(output_dir / 'checkpoint.pt', weights_only=True))


    loss_fn = nn.MSELoss()

    if (output_dir / 'optim.py').is_file():
        optimizer = optim.SGD()
        optimizer.load_state_dict(torch.load(output_dir / 'optim.pt', weights_only=True))

    else:
        optimizer = optim.SGD(model.parameters(), lr = 0.5)

    trained_model = nn_framework.train(model=model,
                                       training_data=train_data,
                                       validation_data=vali_data,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer,
                                       model_dir=output_dir,
                                       num_epochs=1000,
                                       report_step=5,
                                       training_loader_args={'shuffle': False},
                                       validation_loader_args={'shuffle': False})