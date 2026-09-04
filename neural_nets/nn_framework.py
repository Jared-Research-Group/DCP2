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

    # decorator to run handle_kwargs, setup_data after all other __init__ definitions
    @staticmethod
    def initializer(func):
        def wrapped__init__(self, *args, **kwargs):

            func(self, *args, **kwargs)

            self._handle_kwargs(**kwargs) # modify all kwargs passed to init
            self._setup_data()

            return
        return wrapped__init__

    # majority of object initialization. needs to exist outside of __init__ so that we can inherit these definitions without inheriting @initializer with super.__init__
    def _handle_general_constants(self, raw_data_path:Path, output_directory:Path, **kwargs):
            
        self.input_path = raw_data_path # path to data. used by self.preproces_data to generate data.zarr/dataset
        self.output_dir = output_directory # path to dir holding all training data, model results, training analysis, etc.

        self.output_dir.mkdir(exist_ok = True) # create output directory if required

        # model requirements to define in subclass
        self.model_type        = None
        self.loss_fn_type      = torch.nn.MSELoss
        self.otpim_type        = torch.optim.SGD
        self.lr_scheduler_type = torch.optim.lr_scheduler.ReduceLROnPlateau
        self.dataset_type      = None

        # references to objects created later
        self.model    = None
        self.loss_fn  = None
        self.optim    = None
        self.metadata = None

        # default parameters
        self.dataset_splits = [.8, .1, .1]
        self.num_epochs     = 1000
        self.report_step    = 10
        self.lr             = .5 # learning rate

        self.loader_args = {
            'batch_size':         2 ** 10,
            'shuffle':            True,
            'num_workers':        4,
            'persistent_workers': True,
            'pin_memory':         True
        }

        if 'lr' in kwargs: # handle self.lr now so that it can define self.optim_args. All other kwargs handled after all parameters are defined
            self.lr = kwargs['lr']

        self.model_args:        dict[str, typing.Any] = dict()
        self.loss_fn_args:      dict[str, typing.Any] = dict()
        self.optim_args:        dict[str, typing.Any] = {'lr': self.lr}
        self.lr_scheduler_args: dict[str, typing.Any] = {'patience': 50}

        return

    @initializer
    def __init__(self, raw_data_path: Path, output_directory: Path, **kwargs):
        self._handle_general_constants(raw_data_path, output_directory, **kwargs)

        return

    def _handle_kwargs(self, **kwargs):

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
    def _setup_data(self):

        self.data = None

        # detect previously setup data structure
        if (self.output_dir / 'data.zarr').is_dir():

            self.data = zarr.open_group(self.output_dir / 'data.zarr', mode='r')

            # if size of dataset in memory is less than available GPU memory split num_workers ways, then load data directly into memory
            if (self.data['dataset/input'].nbytes + self.data['dataset/target'].nbytes) < constants.TORCH_DEVICE_MEMORY // self.loader_args['num_workers']:
                self.dataset_type = datasets.in_memory_generic
                self.loader_args['pin_memory'] = False # this arg works with non_blocking to parallelize movement of data from CPU to GPU. Not necessary if we move everything to VRAM on load

            else:
                self.dataset_type = datasets.zarr_generic

            dataset = self.dataset_type(self.data['dataset'])

            self.training_dataset   = torch.utils.data.Subset(dataset, np.asarray(self.data.get_array('split_indices/training')[...], dtype=np.int64).tolist())
            self.validation_dataset = torch.utils.data.Subset(dataset, np.asarray(self.data.get_array('split_indices/validation')[...], dtype=np.int64).tolist())
            self.testing_dataset    = torch.utils.data.Subset(dataset, np.asarray(self.data.get_array('split_indices/testing')[...], dtype=np.int64).tolist())

        # new data directory - setup data split accordingly
        else:
            self._preprocess_data()

            assert isinstance(self.data, zarr.Group)

            self.apply_global_normalization()

            # if size of dataset in memory is less than available GPU memory split num_workers ways, then load data directly into memory
            if (self.data['dataset/input'].nbytes + self.data['dataset/target'].nbytes) < constants.TORCH_DEVICE_MEMORY // self.loader_args['num_workers']:
                self.dataset_type = datasets.in_memory_generic
                self.loader_args['pin_memory'] = False # this arg works with non_blocking to parallelize movement of data from CPU to GPU. Not necessary if we move everything to VRAM on load

            else:
                self.dataset_type = datasets.zarr_generic

            original_dataset = self.dataset_type(self.data['dataset'])

            data_splits = torch.utils.data.random_split(original_dataset, self.dataset_splits) # TODO: be cautious of how data is split. Do we see overfitting if training & validation data are overlapping timeseries windows?

            # store indices to reconstruct split data
            split_indices = self.data.create_group('split_indices')
            for name, data in zip(['training', 'validation', 'testing'], data_splits):
                split_indices.create_array(name, 
                                        chunks = (len(data.indices),), 
                                        data   = np.array(data.indices, dtype=np.float64).squeeze()
                                        )

            self.training_dataset, self.validation_dataset, self.testing_dataset = data_splits


        self.training_loader   = torch.utils.data.DataLoader(self.training_dataset, **self.loader_args)
        self.validation_loader = torch.utils.data.DataLoader(self.validation_dataset, **self.loader_args)
        self.testing_loader    = torch.utils.data.DataLoader(self.testing_dataset, **self.loader_args)

    def _preprocess_data(self):
        """
        define in subclass. Should convert raw data to a zarr array of examples. 
        train/vali/test split and dataset initialization are handled downstream

        zarr result should be a group named 'dataset' with arrays 'input' and 'target'.
        store a reference to this group in self.original_dataset.
        """

        raise NotImplementedError

    def apply_global_normalization(self):
        """
        should modify self.data['dataset'] to apply any required global_normalization
        any necessary normalization coefficients should be stored as metadata in self.data['dataset']
        """

        # default behavior: do nothing
        return

    def reverse_global_normalization(self, data):
        """
        should reverse normalization for arbitrary data
        """

        # default behavior: do nothing
        return

    def _setup_training(self):

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

        #with torch.profiler.profile(
        #    activities=[
        #        torch.profiler.ProfilerActivity.CPU,
        #        torch.profiler.ProfilerActivity.CUDA,
        #    ],
        #) as prof:

            self._setup_training()

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

            # setup tqdm progress bar
            pbar = tqdm.trange(self.num_epochs - completed_epochs, ascii = True)
            pbar.set_description(f'lr={self.optim.param_groups[0]['lr']:.2e}')

            for epoch in pbar:

                # training step
                self.model.train()
                for input, target in self.training_loader:

                    # move to GPU if available and not already on GPU
                    if self.dataset_type is not datasets.in_memory_generic:
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

                        if self.dataset_type is not datasets.in_memory_generic:
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

                    pbar.set_description(f'lr={self.optim.param_groups[0]['lr']:.2e}')

                    # save current model state
                    torch.save(self.model.state_dict(), self.output_dir / 'checkpoint.pt')
                    torch.save(self.optim.state_dict(), self.output_dir / 'optim.pt')

                    # save historical loss
                    loss_history.append(local_loss)
                    local_loss = np.zeros(self.report_step)


                    # update plot
                    loss_line.set_xdata(np.arange(loss_history.shape[0]))
                    loss_line.set_ydata(loss_history[:])

                    loss_ax.relim()
                    loss_ax.autoscale_view()
                    loss_fig.canvas.draw_idle()
                    plt.pause(1e-2)

                    loss_fig.savefig(self.output_dir / 'hist.png')

        #        prof.step()

        #prof.key_averages().table(sort_by='cuda_time_total', row_limit=15)
        #prof.export_chrome_trace(str(self.output_dir / 'trace.json'))

            plt.ioff()

            return self.model

    def test(self):

        if self.model is None:
            self._setup_training()
            self.model.to(constants.TORCH_DEVICE)

        self.model.eval()
        with torch.no_grad():

            losses = 0
            samples = 0

            for input, target in self.testing_loader:

                if self.dataset_type is not datasets.in_memory_generic:
                    input  = input .to(constants.TORCH_DEVICE, non_blocking=True)
                    target = target.to(constants.TORCH_DEVICE, non_blocking=True)

                result = self.model(input)
                loss = self.loss_fn(result, target)

                losses += loss.item() * input.size(0)
                samples += input.size(0)


            loss = losses / samples

            print(f'loss={loss:2e}')