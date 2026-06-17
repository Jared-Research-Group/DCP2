# this file contains helper functions to perform computations and store calcualted values

import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd
import scipy.stats as stats
import os
import sympy as sp
import json
from tqdm import tqdm
from pathlib import Path
import shutil

import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog

#############################

# Stats helper functions

# reworking statistics to use pd rolling method
def applyRollingOperation(data, window_length, op, **args):

    original_type = type(data)
    data = pd.Series(data)

    # error checking input
    if np.isnan(data).any():
        raise ValueError(op.__name__ + ': Array has NaN')

    if window_length <= 1:
        raise ValueError(op.__name__ + ': Average window too small')
    
    result = data.rolling(window_length, closed='both').op(*args)

    return original_type(result)

def getRollingAvg(data, avg_scale=1000):
    return applyRollingOperation(data, avg_scale, np.mean)

def getRollingStdDev(data, sd_scale=5000):
    return applyRollingOperation(data, sd_scale, np.std)

def getRollingSkew(data, sk_scale):
    return applyRollingOperation(data, sk_scale, stats.skew)

def getRollingKurtosis(data, k_scale):
    return applyRollingOperation(data, k_scale, stats.kurtosis)

#############################

# takes an array and a limit value and returns the start:stop indices that bound  (array's value) > testLimit
def getStartStop(testVal, testLimit = 1):

    startTime = 0
    for t, v in enumerate(testVal):
        if v > testLimit: 
            startTime = t
            break

    stopTime = 0
    for t, v in enumerate(testVal[1:]):                                     # changed this to [1:], keep an eye on this if issues arise
        if t < startTime:
            continue

        if testVal[t - 1] >=  testLimit and v < testLimit:
            stopTime = t

    if stopTime == 0: stopTime = len(testVal)

    return startTime, stopTime

def quickPlot(data, s=0.005):
    plt.style.use('_mpl-gallery')
    fig, ax = plt.subplots(len(data), 1, constrained_layout=True)

    if len(data) > 1:
        for j, d in enumerate(data):
            for i, val in enumerate(d):
                ax[i][j].scatter(val[0], val[1], s=s)
    else:
        for i, val in enumerate(data[0]):
            ax[i].scatter(val[0], val[1], s=s)

    return

def dfToCsv(df, f):
    df.to_csv(f, index=False)

def dfHasColumn(df, id):
    if id in df.columns: return True
    return False

def csvHasColumn(f, id):
    if id in pd.read_csv(f, nrows=1): return True
    return False

def selectFolder(title='Select Top-Level Folder'):
    init_dir = os.path.expanduser('~')
    if os.access(init_dir + '/Data', os.R_OK):
        init_dir += '/Data'

    root = tk.Tk()
    root.wm_attributes('-topmost', 1)
    root.withdraw()

    path = filedialog.askdirectory(
        title=title,
        initialdir = init_dir,
        parent=root
    )

    return path

def setup_directory_structure(dir, input_file, output_files, **kwargs):
    """ Generalize input/output file/directory setup for initial postprocess steps

    Args:
        dir (str | pathlib.Path): 'data_collection_' directory
        input_file (str): name of input file / directory within dir. Overridden to arbitrary location with kwarg 'input_path'
        output_files (list(str)): list of names of output files / directories within dir (arbitrary number of outputs). Overridden to arbitrary list of paths with kwarg 'output_paths'
    """

    # setup input/output locations
    dir = Path(dir)

    # create safe raw data folder and store original data there
    if not os.access(dir / 'raw_data', os.R_OK):
        os.mkdir(dir / 'raw_data')

    # define input path location with kwarg override for arbitrary location
    input_path = dir / input_file
    if 'input_path' in kwargs: input_path = Path(kwargs['input_path'])

    # define path to which input file is stored safely
    new_input_name = input_path.stem + '__raw' + input_path.suffix

    new_input_path = dir / 'raw_data' /  new_input_name
    if 'store_path' in kwargs: new_input_path = Path(kwargs['store_path'])

    # define output path locations with kwarg override for arbitrary location
    output_paths = [dir / filename for filename in output_files]
    if 'output_paths' in kwargs: output_paths = [Path(output_path) for output_path in kwargs['output_paths']]

    # make a safe copy of raw input data in the data collection folder

    if not os.access(new_input_path, os.F_OK):
        shutil.move(input_path, new_input_path)

    return new_input_path, output_paths

#############################

# FLIR helper functions

# data must be np.array!
def flirConversion(data, model):
    temps = np.zeros(data.shape)

    count = 1
    for i, intensity in tqdm(np.ndenumerate(data)):
        temps[i] = model(float(intensity)) - 273.15

        count += 1
    return temps

class CaseSelectionDialog(simpledialog.Dialog):
    def body(self, master):
        tk.Label(master, text="What temperature range is FLIR data in?").grid(row=0, column=0, sticky="w")

    def buttonbox(self):
        box = tk.Frame(self)

        tk.Button(box, text='High', width=10, command=self.high).pack(side="left", padx=5, pady=5)
        tk.Button(box, text='Low', width=10, command=self.low).pack(side="left", padx=5, pady=5)
        tk.Button(box, text='Cancel', width=10, command=self.cancel).pack(side="left", padx=5, pady=5)

        self.bind("<Return>", self.cancel)
        self.bind("<Escape>", self.cancel)
        box.pack()

    def high(self, event=None):
        self.result = 1
        self.cancel()

    def low(self, event=None):
        self.result = 0
        self.cancel()

def ask_case():
    root = tk.Tk()
    root.withdraw()

    dialog = CaseSelectionDialog(root, title='Case Selection')
    return dialog.result

def get_FLIR_model(d_in):
    import pysr

    d_in = Path(d_in)

    with open(d_in / 'FLIR_Variables.json', 'r') as json_file:
        params = json.load(json_file)

    if 'case' in params:
        case = int(params['case'])
    else:

        case = ask_case()                   # dialog needs to close after button press. json assignment doesnt work, and need to rewrite file.
        params["case"] = str(case)

        with open(d_in / 'FLIR_Variables.json', 'w') as json_file:
            json.dump(params, json_file, indent=2)

    x = sp.symbols('FLIR_Intensity')

    if case == 1:
        high_fit = pysr.PySRRegressor().from_file(run_directory=os.getcwd() + '/FLIR_fits/High', model_selection='best')
        return sp.lambdify(x, high_fit.sympy(), modules='numpy')
    else:
        low_fit = pysr.PySRRegressor().from_file(run_directory=os.getcwd() + '/FLIR_fits/Low', model_selection='best')
        return sp.lambdify(x, low_fit.sympy(), modules='numpy')