# generic imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import zarr
from pathlib import Path
import typing
import asyncio

# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# project imports
import constants

# TODO: be cautious of how data is split. Do we see overfitting if training & validation data are overlapping timeseries windows?
def split_data(dataset: torch.utils.data.Dataset, splits: list = [.8, .1, .1]) -> list:

    return torch.utils.data.random_split(dataset, splits) # training, validation, testing (torch.utils.data.Datasets)

# generic DataLoader init. Allows arbitrary argument passthrough
def setup_dataloader(dataset: torch.utils.data.Dataset, special_args: dict = dict()) -> torch.utils.data.DataLoader:

    loader_args = {
        'batch_size':         2 ** 10,
        'shuffle':            True,
        'num_workers':        4,
        'persistent_workers': True,
        'pin_memory':         True
    }

    for key, val in special_args.items():

        if key in loader_args:
            loader_args[key] = val

        else:
            raise Exception(f'torch.utils.data.DataLoader has no argument "{key}"')

    return torch.utils.data.DataLoader(dataset, **loader_args)

# training loop. #TODO: add kwargs for further customization (as necessary)
def train(model: torch.nn.Module, 
          training_data: torch.utils.data.Dataset, 
          validation_data: torch.utils.data.Dataset,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          model_dir: Path,
          num_epochs: int = 1000,
          report_step: int = 100,
          **kwargs
          ) -> torch.nn.Module:

    # setup training, validation data loaders. Arguments passable via kwargs.
    training_loader_args = kwargs['training_loader_args'] if 'training_loader_args' in kwargs else dict()
    training_loader = setup_dataloader(training_data, training_loader_args)

    validation_loader_args = kwargs['validation_loader_args'] if 'validation_loader_args' in kwargs else dict()
    validation_loader = setup_dataloader(validation_data, validation_loader_args)

    # load required metadata if we are continuing from an existing checkpoint
    if (model_dir / 'metadata.zarr').is_dir():

        metadata = zarr.open_group(model_dir / 'metadata.zarr', mode='r+')
        loss_history = typing.cast(zarr.Array, metadata['loss_history'])
        completed_epochs = loss_history.shape[0]

    # init metadata if we are training a new model
    else:

        metadata = zarr.create_group(model_dir / 'metadata.zarr')
        loss_history = metadata.create_array('loss_history', shape = (0,), chunks = (1000,), dtype = 'float64')
        completed_epochs = 0

    # move to GPU if available
    model.to(constants.TORCH_DEVICE)

    local_loss = np.zeros(report_step)

    loss_fig, loss_ax = plt.subplots(1, 1, layout='constrained')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel(f'{loss_fn.__class__.__name__}')
    loss_ax.set_title('LSTM Loss History')

    loss_line, = loss_ax.semilogy(np.arange(loss_history.shape[0]), loss_history[:], color='blue')

    plt.ion()
    plt.show()
    plt.pause(1e-2)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=800)

    for epoch in tqdm(range(num_epochs - completed_epochs)):

        # training step
        model.train()
        for input, target in (training_loader):

            # move to GPU if available
            input  = input .to(constants.TORCH_DEVICE, non_blocking=True)
            target = target.to(constants.TORCH_DEVICE, non_blocking=True)

            model.zero_grad()
            result = model(input)
            loss = loss_fn(result, target)
            loss.backward()
            optimizer.step()

        # validation step TODO: modify learning rate based on loss / loss improvement
        model.eval()
        with torch.no_grad():

            # doing a sort-of batched (weighted) averaging to compute validation loss.

            losses = 0
            samples = 0

            for input, target in validation_loader:

                input  = input .to(constants.TORCH_DEVICE, non_blocking=True)
                target = target.to(constants.TORCH_DEVICE, non_blocking=True)

                result = model(input)
                loss = loss_fn(result, target)

                losses += loss.item() * input.size(0)
                samples += input.size(0)


            local_loss[epoch % report_step] = losses / samples

        scheduler.step(local_loss[epoch % report_step])

        # update plots every 10 epochs
        if epoch % report_step == report_step - 1:

            loss_history.append(local_loss)
            local_loss = np.zeros(report_step)

            loss_line.set_xdata(np.arange(loss_history.shape[0]))
            loss_line.set_ydata(loss_history[:])

            loss_ax.relim()

            loss_ax.autoscale_view()

            loss_fig.canvas.draw_idle()
            plt.pause(1e-2)

            loss_fig.savefig(model_dir / 'hist.png')

            torch.save(model.state_dict(), model_dir / 'checkpoint.pt')
            torch.save(optimizer.state_dict(), model_dir / 'optim.pt')

            print(f'Learning rate: {optimizer.param_groups[0]['lr']}')

    return model