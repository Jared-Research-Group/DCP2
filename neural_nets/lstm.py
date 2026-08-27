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
hidden_dim = 32

output_dir = Path(r"C:\Users\wwerner4\Documents\data\nn_test")

if __name__ == '__main__':

    data_dir = Path(helper_functions.selectFolder())

    data = datasets.ThermalSequenceDataset(data_dir,
                                           input_seq_len=input_length,
                                           output_seq_len=output_length, 
                                           step=step)

    train_data, vali_data, test_data = nn_framework.split_data(data)

    model = models.ThermalLSTM(hidden_dim=hidden_dim,
                               input_sequence_length=input_length,
                               output_sequence_length=output_length)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr = 1e-5)

    trained_model = nn_framework.train(model=model,
                                       training_data=train_data,
                                       validation_data=vali_data,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer,
                                       model_dir=output_dir,
                                       num_epochs=5000)