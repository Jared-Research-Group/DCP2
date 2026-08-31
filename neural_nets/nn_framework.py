# generic imports
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import zarr
from pathlib import Path
import typing

# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# project imports
import helper_functions
import constants
import datasets

logger = helper_functions.setup_logger(__name__)

class NeuralNet:

    def __init__(self, raw_data_path: Path, output_directory: Path, **kwargs):

        self.input_path = raw_data_path
        self.output_dir = output_directory

        # create output directory if required
        self.output_dir.mkdir(exist_ok = True)

        # define in subclass
        self.model_type        = None
        self.loss_fn_type      = torch.nn.MSELoss
        self.otpim_type        = torch.optim.SGD
        self.lr_scheduler_type = torch.optim.lr_scheduler.ReduceLROnPlateau

        self.model    = None
        self.loss_fn  = None
        self.optim    = None
        self.metadata = None

        # default parameters (too many for named arguments, so these can be modified by **kwargs)
        self.dataset_splits = [.8, .1, .1]
        self.num_epochs     = 1000
        self.report_step    = 10
        self.lr             = .5

        self.loader_args = {
            'batch_size':         2 ** 10,
            'shuffle':            True,
            'num_workers':        4,
            'persistent_workers': True,
            'pin_memory':         True
        }

        self.handle_kwargs(**kwargs)

        self.model_args:        dict[str, typing.Any] = dict()
        self.loss_fn_args:      dict[str, typing.Any] = dict()
        self.optim_args:        dict[str, typing.Any] = {'lr': self.lr}
        self.lr_scheduler_args: dict[str, typing.Any] = {'patience': 50}

        self.setup_data()

        return

    def handle_kwargs(self, **kwargs):

        # parse kwarg overrides of members
        for key, arg in kwargs.items():

            # don't modify any members that we don't have a default for (non-valid parameters)
            if hasattr(self, key):

                # if we are modifying a member dict, iterate over kwarg dict with modifications
                if key in ['loader_args']:
                    
                    for k, a in arg.items():

                        if k in getattr(self, key): # if key is valid, modify it
                            getattr(self, key)[k] = a

                # normal parameter modification
                else:
                    setattr(self, key, arg)

            else:
                logger.warning(f'{key} is not a valid attribute of {self.__class__.__name__}')

        return

    # handle datasets + data loaders
    def setup_data(self):

        self.data = None

        # detect previously setup data structure
        if (self.output_dir / 'data.zarr').is_dir():

            self.data = zarr.open_group(self.output_dir / 'data.zarr', mode='r')

            dataset = datasets.zarr_generic(self.data['dataset'])

            self.training_dataset   = torch.utils.data.Subset(dataset, np.asarray(self.data.get_array('split_indices/training')[...], dtype=np.int64).tolist())
            self.validation_dataset = torch.utils.data.Subset(dataset, np.asarray(self.data.get_array('split_indices/validation')[...], dtype=np.int64).tolist())
            self.testing_dataset    = torch.utils.data.Subset(dataset, np.asarray(self.data.get_array('split_indices/testing')[...], dtype=np.int64).tolist())

        # new data directory - setup data split accordingly
        else:
            self.preprocess_data()

            assert self.data is zarr.Group

            self.apply_global_normalization()
            original_dataset = datasets.zarr_generic(self.data['dataset'])

            data_splits = torch.utils.data.random_split(original_dataset, self.dataset_splits) # TODO: be cautious of how data is split. Do we see overfitting if training & validation data are overlapping timeseries windows?

            # store indices to reconstruct split data
            split_indices = self.data.create_group('split_indices')
            for name, data in zip(['training', 'validation', 'testing'], data_splits):
                split_indices.create_array(name, 
                                        shape  = (len(data.indices),), 
                                        chunks = (len(data.indices),), 
                                        data   = data.indices,
                                        dtype  = 'int64')

            self.training_dataset, self.validation_dataset, self.testing_dataset = data_splits


        self.training_loader   = torch.utils.data.DataLoader(self.training_dataset, **self.loader_args)
        self.validation_loader = torch.utils.data.DataLoader(self.validation_dataset, **self.loader_args)
        self.testing_loader    = torch.utils.data.DataLoader(self.testing_dataset, **self.loader_args)

    def preprocess_data(self):
        """
        define in subclass. Should convert raw data to a zarr array of examples. 
        train/vali/test split and dataset initialization are handled downstream

        zarr result should be a group named 'dataset' with arrays 'input' and 'target'.
        store a reference to this group in self.original_dataset.
        """

        raise NotImplementedError

    def apply_global_normalization(self):
        """
        should modify self.data to apply any required global_normalization
        """

        # default behavior: do nothing
        return

    def setup_training(self):

        assert self.model_type is not None

        self.model = self.model_type(**self.model_args)
        if (self.output_dir / 'checkpoint.pt').is_file():
            self.model.load_state_dict(torch.load(self.output_dir / 'checkpoint.pt', weights_only = True))

        self.loss_fn = self.loss_fn_type(**self.loss_fn_args)

        self.optim = self.otpim_type(self.model.parameters(), **self.optim_args)
        if (self.output_dir / 'optim.pt').is_file():
            self.optim.load_state_dict(torch.load(self.output_dir / 'optim.pt', weights_only = True))

        self.lr_scheduler = self.lr_scheduler_type(self.optim, **self.lr_scheduler_args)

        # load required metadata if we are continuing from an existing checkpoint
        if (self.output_dir / 'metadata.zarr').is_dir():
            self.metadata = zarr.open_group(self.output_dir / 'metadata.zarr', mode='r+')
            

        # init metadata if we are training a new model
        else:
            self.metadata = zarr.create_group(self.output_dir / 'metadata.zarr')
            self.metadata.create_array('loss_history', shape = (0,), chunks = (1024,), dtype = 'float64')

        return

    # training loop. #TODO: add kwargs for further customization (as necessary)
    def train(self) -> torch.nn.Module:

        self.setup_training()

        assert self.metadata is not None
        assert self.loss_fn  is not None
        assert self.model    is not None
        assert self.optim    is not None

        loss_history = typing.cast(zarr.Array, self.metadata['loss_history'])
        completed_epochs = loss_history.shape[0]

        # move to GPU if available
        self.model.to(constants.TORCH_DEVICE)

        local_loss = np.zeros(self.report_step)

        # setup live loss visualization
        loss_fig, loss_ax = plt.subplots(1, 1, layout='constrained')
        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel(f'{self.loss_fn.__class__.__name__}')
        loss_ax.set_title('LSTM Loss History')

        loss_line, = loss_ax.semilogy(np.arange(loss_history.shape[0]), loss_history[:], color='blue')

        plt.ion()
        plt.show()
        plt.pause(1e-1)

        for epoch in tqdm.trange(self.num_epochs - completed_epochs, ascii = True, desc = f'lr={self.optim.param_groups[0]['lr']}'):

            # training step
            self.model.train()
            for input, target in self.training_loader:

                # move to GPU if available
                input  = input .to(constants.TORCH_DEVICE, non_blocking=True)
                target = target.to(constants.TORCH_DEVICE, non_blocking=True)

                self.model.zero_grad()
                result = self.model(input)
                loss = self.loss_fn(result, target)
                loss.backward()
                self.optim.step()

            # validation step TODO: modify learning rate based on loss / loss improvement
            self.model.eval()
            with torch.no_grad():

                losses = 0
                samples = 0

                for input, target in self.validation_loader:

                    input  = input .to(constants.TORCH_DEVICE, non_blocking=True)
                    target = target.to(constants.TORCH_DEVICE, non_blocking=True)

                    result = self.model(input)
                    loss = self.loss_fn(result, target)

                    losses += loss.item() * input.size(0)
                    samples += input.size(0)


                local_loss[epoch % self.report_step] = losses / samples

            self.lr_scheduler.step(local_loss[epoch % self.report_step])

            # update plots every 10 epochs
            if epoch % self.report_step == self.report_step - 1 or epoch == self.num_epochs - 1:

                loss_history.append(local_loss)
                local_loss = np.zeros(self.report_step)

                loss_line.set_xdata(np.arange(loss_history.shape[0]))
                loss_line.set_ydata(loss_history[:])

                loss_ax.relim()
                loss_ax.autoscale_view()

                loss_fig.canvas.draw_idle()
                plt.pause(1e-2)

                loss_fig.savefig(self.output_dir / 'hist.png')

                torch.save(self.model.state_dict(), self.output_dir / 'checkpoint.pt')
                torch.save(self.optim.state_dict(), self.output_dir / 'optim.pt')

        return self.model