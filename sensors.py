import math
import zarr
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import datetime
import xml.etree.ElementTree as ET
import tqdm
from PIL import Image

import helper_functions

logger = helper_functions.setup_logger(__name__)

# parent class for all DCP2 sensors
class BaseSensor:

    @staticmethod
    def initializer(func):
        def wrapped__init__(self, *args, **kwargs):

            func (self, *args, **kwargs)

            self._link_data()

            return

        return wrapped__init__

    def __init__(self, input_path: Path):

        # standard sensor info. potentially overwritten by info in zarr store
        self.name             = ''
        self.path             = input_path
        self.acquisition_rate = None
        self.shape            = tuple() # shape of a single sample. e.g. 64x64 image = (,64,64); 1D timeseries = tuple(); 2D timeseries = (2,) 
        self.dtype            = np.float64
        self.columns          = tuple()

        self.group            = None
        self.time_chunk       = None
        self.data_chunk       = None

        self.data = None
        self.time = None

    def _link_data(self):
        # if the input path is a zarr group, then extract information from the sensor's subgroup, if available
        if self.path.suffix == '.zarr':

            try:
                zarr_root = zarr.open_group(self.path)

                self.group = zarr_root[self.name]
                self.data = self.group['data']
                self.time = self.group['time']

                self.acquisition_rate = self.group.attrs['acquisition_rate']
                self.columns        = self.group.attrs['columns']
                self.dtype = self.data.dtype
                if len(self.data.shape) > 1: self.shape = self.data.shape[1:]
                
            except Exception as e:

                logger.info(str(e))

        else:

            zarr_root = zarr.open_group(self.path.parent / (self.path.parent.name + '.zarr'))
            self.group = zarr_root.create_group(self.name)

            self._build_arrays()

            self._convert_data()

            self.path = zarr_root.path

    def _get_chunk_sizes(self):

        assert isinstance(self.acquisition_rate, (float, int))
        
        # chunk to 1MB chunks (recommendation of zarr docs)
        self.time_chunk = (math.ceil(10 ** 6 / (math.prod(self.shape) * 8)),)     # TODO: what type are we using? Always float/int64?
        
        # if we are handling one/multiple 1D timeseries columns, chunk down each column separately (allow for selective column reads)
        if len(self.shape) < 1:
            self.data_chunk = self.time_chunk

        elif len(self.shape) == 1:
            self.data_chunk = self.time_chunk + (1,)
        
        # if we are handling 2D+ data, don't bother to chunk in dimensions other than time
        else:
            self.data_chunk = self.time_chunk + self.shape

        logger.info(f'{self.name}: time_chunk: {self.time_chunk}, data_chunk: {self.data_chunk}')

    def _build_arrays(self):

        self._get_chunk_sizes()

        self.time = self.group.create_array(name = 'time', 
                                            shape = (0,), 
                                            chunks = self.time_chunk, 
                                            dtype = np.uint64)
        
        self.data = self.group.create_array(name = 'data', 
                                            shape = (0,) + self.shape, 
                                            chunks = self.data_chunk, 
                                            dtype = self.dtype)

        self.group.attrs['acquisition_rate'] = self.acquisition_rate
        self.group.attrs['columns']       = self.columns

    def display_data(self, flag_sparse_plot = True):

        data_length = len(self.time[:])

        # get sparse sampling of data indices for sparse plotting of data if required
        if data_length > 1e5 and flag_sparse_plot:
            idx = np.arange(1e5, dtype = int) * int(data_length / 1e5)
        else:
            idx = np.arange(data_length, dtype = int)

        # convert zarr data to df and plot each variable
        df = pd.DataFrame({'Time': self.time[:][idx] - self.time[:][0]})
        df['Time'] /= 1e9

        if self.shape == tuple(): data = [self.data[:][idx].T]
        else:                     data = self.data[:][idx].T
                          
        for i, row in enumerate(data):
            df[self.columns[i]] = row

            fig = px.scatter(data_frame = df, x = 'Time', y = self.columns[i])
            fig.update_traces(marker = {'size': 4 - (len(df['Time']) * (2 / 1e5))})
            fig.show()

class LEMBox(BaseSensor):

    @BaseSensor.initializer
    def __init__(self, input_dir: Path):

        super(LEMBox, self).__init__(input_dir)

        self.name = 'LEMBox'
        self.acquisition_rate = 20e3
        self.shape = (2,)
        self.columns = ('Voltage(V)', 'Curent(A)')

    def _convert_data(self):

        df = pd.read_csv(self.path).dropna()
        df['Timestamp'] = pd.to_datetime(df['Timestamp'].str[:-4])
        df['Timestamp'] = df['Timestamp'].astype('int64')

        time = df['Timestamp'].to_numpy()
        time = (np.arange(len(df['Timestamp'])) / self.acquisition_rate) * 1e9 + time[0]

        self.time.append(time)
        del time

        voltage = df['Voltage(V)'].to_numpy() * 10
        current = df['Current(A)'].to_numpy() * 100

        data = np.column_stack((voltage, current))
        del voltage, current

        self.data.append(data)

        return

class ThermocoupleDAQ(BaseSensor):

    @BaseSensor.initializer
    def __init__(self, input_dir: Path):

        super(ThermocoupleDAQ, self).__init__(input_dir)

        self.name = 'ThermocoupleDAQ'
        self.acquisition_rate = 3.5
        self.shape = (4,)
        self.columns = ('TC0', 'TC1', 'TC2', 'TC3')

    def _convert_data(self):

        df = pd.read_csv(self.path, encoding = 'cp1252').dropna()
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Timestamp'] = df['Timestamp'].astype('int64')

        time = df['Timestamp'].to_numpy()
        self.time.append(time)
        del time

        ch_0 = df.iloc[:, 2].to_numpy()
        ch_1 = df.iloc[:, 3].to_numpy()
        ch_2 = df.iloc[:, 4].to_numpy()
        ch_3 = df.iloc[:, 5].to_numpy()

        data = np.column_stack((ch_0, ch_1, ch_2, ch_3))
        del ch_0, ch_1, ch_2, ch_3

        self.data.append(data)

class Microphone(BaseSensor):

    @BaseSensor.initializer
    def __init__(self, input_dir: Path):

        super(Microphone, self).__init__(input_dir)

        self.name = 'Microphone'
        self.acquisition_rate = 48e3

        self.columns = ('Amplitude',)

    def _convert_data(self):

        df = pd.read_csv(self.path, skiprows = 1).dropna()
        df['Absolute Time'] = pd.to_datetime(df['Absolute Time'])
        df['Absolute Time'] = df['Absolute Time'].astype('int64')

        time = df['Absolute Time'].to_numpy()
        time = (np.arange(len(df['Absolute Time'])) / self.acquisition_rate) * 1e9 + time[0]    
        self.time.append(time)
        del time

        amp = df['Amplitude']
        self.data.append(amp)

class RSI(BaseSensor):

    @BaseSensor.initializer
    def __init__(self, input_dir: Path):

        super(RSI, self).__init__(input_dir)

        self.name = 'RSI'
        self.acquisition_rate = 250

        self.columns = ('JSON',)

    def _convert_data(self):

        with open(self.path, 'r') as f:

            next(f) # skip file header
            prev_IPOC = '' # older files have duplicate lines from RSI (we were sampling faster than the robot was updating)

            times = []
            data = []

            for line in f:
                if line.strip(): # remove whitespace

                    parts = line.strip().split('|') # extract absolute, relative times + data
                    if len(parts) != 3:
                        return None

                    system_time, relative_time, xml_data = parts

                    root = ET.fromstring(xml_data)
                    parsed_data = self.parse_xml(root)

                    # skip duplicate xml lists
                    if parsed_data['IPOC'] == prev_IPOC:
                        continue

                    # re-initialize self.data zarr array if we are seeing the first sample
                    if prev_IPOC == '':

                        # extract column names, build 
                        self.columns = tuple(name for name in parsed_data.keys() if name != 'Rob_Type')
                        self.shape = (len(self.columns),)
                        self.columns = ('data',)

                        zarr_root = zarr.open_group(self.path.parent / (self.path.parent.name + '.zarr'))
                        self.group = zarr_root[self.name]

                        del self.group['data']
                        del self.group['time']
                        
                        self._build_arrays()

                    # store this sample
                    times.append(int(datetime.datetime.strptime(system_time, '%Y-%m-%d %H:%M:%S.%f').timestamp() * 10**9)) # get absolute time
                    data.append([parsed_data[key] for key in parsed_data.keys() if key != 'Rob_Type'])
                    prev_IPOC = parsed_data['IPOC']

            self.time.append(np.array(times, dtype = np.int64))
            self.data.append(np.array(data))


    # extract data from xml string. returns dict
    def parse_xml(self, element: str) -> dict:
        data = {}

        for key, value in element.attrib.items():
            try:
                data[f"{element.tag}_{key}"] = float(value)
            except ValueError:
                data[f"{element.tag}_{key}"] = value

        if element.text and element.text.strip():
            try:
                data[element.tag] = float(element.text)
            except ValueError:
                data[element.tag] = element.text.strip()

        for child in element:
            data.update(self.parse_xml(child))

        return data

class FLIR(BaseSensor):
        
    @BaseSensor.initializer
    def __init__(self, input_dir: Path):

        super(FLIR, self).__init__(input_dir)

        self.name = 'FLIR'
        self.acquisition_rate = 30
        self.shape = (348, 464)
        self.columns = ('FLIR')

    def _convert_data(self):

        frames = [np.load(file, allow_pickle = True).item() for file in self.path.iterdir() if file.suffix != '.json']

        for frame in frames: frame['timestamp'] = datetime.datetime.strptime(frame['timestamp'], '%Y-%m-%d %H:%M:%S.%f').timestamp() * 10**9

        frames.sort(key = lambda x: x['timestamp'])

        time = np.array([frame['timestamp'] for frame in frames])
        self.time.append(time)
        del time

        data = np.array([frame['frame'] for frame in frames])
        self.data.append(data)

    def display_data(self, flag_sparse_plot = True):

        data_length = len(self.time[:])

        # get sparse sampling of data indices for sparse plotting of data if required
        if data_length > 3e2 and flag_sparse_plot:
            idx = np.arange(3e2, dtype = int) * int(data_length / 3e2)
        else:
            idx = np.arange(data_length, dtype = int)

        fig = px.imshow(self.data[:][idx], animation_frame=0)#, labels=self.time[:])
        fig.show()


class Xiris(BaseSensor):

    @BaseSensor.initializer
    def __init__(self, input_dir: Path):

        super(Xiris, self).__init__(input_dir)

        self.name = 'Xiris'
        self.acquisition_rate = 50
        self.shape = (512, 640)
        self.columns = ('Xiris')

    def _build_arrays(self):

        super(Xiris, self)._build_arrays()

        self.meta = self.group.create_array(name = 'meta',
                                            shape = (0,),
                                                     chunks = self.time_chunk,
                                                     dtype = 'U180')

    @staticmethod
    def convertFrame(f):
        #Efficiently read raw frame data using NumPy bulk read instead of pixel-by-pixel loop
 
        time = int(int(f.stem) * 1e3)

        with open(f, "rb") as frame:
            frame.read(8)  # Skip header
            left = int.from_bytes(frame.read(4), 'little')
            top = int.from_bytes(frame.read(4), 'little')
            right = int.from_bytes(frame.read(4), 'little')
            bottom = int.from_bytes(frame.read(4), 'little')
    
            width = 640
            height = bottom - top
    
            frame.read(32)  # Skip remaining header
        
            img = Image.frombytes('I;16', [width, height], frame.read(2 * width * height))
            intensity = np.asarray(img) * 2**2

            meta = frame.read()
    
        return time, intensity, meta
 

    def _convert_data(self):

        frames = [frame for frame in (self.path / 'raw').iterdir()]

        with open(frames[0], 'rb') as f:
            f.read(8)

            left = int.from_bytes(f.read(4), 'little')
            top = int.from_bytes(f.read(4), 'little')
            right = int.from_bytes(f.read(4), 'little')
            bottom = int.from_bytes(f.read(4), 'little')

            self.group.attrs['AoI'] = [left, right, top, bottom]

        for frame in tqdm.tqdm(frames):

            time, img, meta = self.convertFrame(frame)
            self.time.append([time])
            self.data.append([img])
            self.meta.append([meta])

    def display_data(self, flag_sparse_plot = True):

        data_length = len(self.time[:])
        sparse_length = 15

        # get sparse sampling of data indices for sparse plotting of data if required
        if data_length > sparse_length and flag_sparse_plot:
            idx = np.arange(sparse_length, dtype = int) * int(data_length / sparse_length)
        else:
            idx = np.arange(data_length, dtype = int)

        fig = px.imshow(self.data[:][idx], animation_frame=0)#, labels=self.time[:])
        fig.show()


