import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
import time
import yaml
from pathlib import Path

build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'build', 'lib.win-amd64-cpython-310')
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)
    
import pysr
import sympy
from sympy.abc import x
from datetime import timedelta

from thermography import getPixels, getFrameData, getHotFrame
from data_manipulation import selectFolder, dfToCsv
from batch_process import dataSearch

validation_datasets = []

# read FLIR temp data from selected pixels & frames, add this thermal data to aligned data csv. returns + saves windowed dataframe
def getCalData(dir, recalc_temps=False, window_length=10, needTimes=False):
    # if data isn't stored, calculate it

    dir = Path(dir)
    temp_regime = dir.name

    if (window_length != -1 and not os.access(dir.parent / (temp_regime + '.csv'), os.R_OK)) or (window_length == -1 and not os.access(dir.parent / (temp_regime + '_unwindowed.csv'), os.R_OK)) or recalc_temps:
        top_level_dir = dir.parents[2]

        df = pd.read_csv(dir / 'aligned_data.csv')

        # we need to window data in time to cut out obfuscation of wall in video, thermal dropoff towards end of data collection
        if window_length != -1:

            start_index = 0

            df = df.loc[(df['time'] > df['time'][start_index]) & (df['time'] < (df['time'][start_index] + window_length))]
            df.reset_index(inplace=True)

            # save windowed data to new .csv for later manipulation
            df.to_csv(dir.parent / (temp_regime + '.csv'), index=False)

        else:
            df.to_csv(dir.parent / (temp_regime + '_unwindowed.csv'), index=False)

    # if windowed data is already saved, just load it from the file
    else:
        if window_length != -1:
            df = pd.read_csv(dir.parent / (temp_regime + '.csv'))
        else:
            df = pd.read_csv(dir.parent / (temp_regime + '_unwindowed.csv'))

    channel_num = 3

    if needTimes:
        return (df['Channel_' + str(channel_num) + '(°C)'], df['Pyrometer_Temp(°C)'], temp_regime, df['time'])
    else:
        return (df['Channel_' + str(channel_num) + '(°C)'], df['Pyrometer_Temp(°C)'], temp_regime)

# requires input arrays nested in tuples, allows for multiple datasets at once
def plotCalCurve(data, fit=None, vali_data=None):

    plt.figure(layout='constrained', figsize=[4, 3])

    pyro_temp, tc_temp, temp_regime = data

    #tc_temp -= 273.15        # convert back to °C
            
    series_fit = fit

    fit_temp = np.linspace(0, 20000, 20000)

    if type(series_fit) == pysr.PySRRegressor:
        #fit_vals = series_fit.predict(fit_temp.reshape(-1, 1))
        fit_str = str(series_fit.sympy(7))
        print(series_fit)
        
        
        
        x = sympy.Symbol('Pyrometer_Temperature')
        
        # 11, 10 bad case
        fit_alt    = sympy.lambdify(x, series_fit.sympy(11))
        fit_alt2    = sympy.lambdify(x, series_fit.sympy(10))
        
        series_fit = sympy.lambdify(x, series_fit.sympy(7))
        fit_vals = series_fit(fit_temp)
        print(fit_str)
    else:
        fit_str = str(series_fit)
        series_fit = sympy.lambdify(x, series_fit, "numpy")
        fit_vals = series_fit(fit_temp)


    #fit_vals -= 273.15  # convert back to °C
    alt_fit_vals = fit_alt(fit_temp)
    alt_fit_vals2 = fit_alt2(fit_temp)

    plt.plot(fit_temp, fit_vals, color='orange', label=(str('Calibration Curve')), linewidth=4, alpha=.4)
    plt.plot(fit_temp, alt_fit_vals, color='red', label=(str('1st Alternate Calibration Curve')), linewidth=4, alpha=.4)
    plt.plot(fit_temp, alt_fit_vals2, color='green', label=(str('2nd Alternate Calibration Curve')), linewidth=4, alpha=.4)

    plt.scatter(pyro_temp, tc_temp, color='blue', label='Calibration Training Data', s=10 )  
            
    if vali_data is not None:
        for i, data in enumerate(vali_data):
            vali_intensity, vali_temp, vali_name = data
            vali_temp -= 273.15

            #plt.scatter(vali_intensity, vali_temp, label='Validation Curve ' + str(i), s=0.05)
            if i == 0:
                plt.scatter(vali_intensity, vali_temp, label='Validation Curves', s=0.075)
            else:
                plt.scatter(vali_intensity, vali_temp, label='_Validation Curve', s=0.075)
    
    if len(data) > 1 or fit is not None or vali_data is not None:
        leg = plt.legend(fontsize=8)
        for l in leg.legend_handles:
            l._sizes = [80]
    plt.xlabel('Raw Pyrometer Temperature (°C)', fontsize=8)
    plt.ylabel('Thermocouple\nTemperature (°C)', fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    
    plt.grid(True)

    plt.ylim(-50, 500)
    plt.xlim(0, 1000)
    #plt.ylim(0, 600)
    #plt.xlim(0, 40000)
    plt.show()

    return

def combineData(dir, inclusions, validation=False, force_update=False):
    dir = Path(dir)

    if not os.access(dir / 'Combined_Calibration_Data.csv', os.R_OK) or force_update:
        df = pd.DataFrame({'pyro_temp(°C)':pd.Series(dtype='float64'), 'tc_temp(°C)':pd.Series(dtype='float64'), 'experiment':pd.Series(dtype='str')})

        def getDataSubset(d):
            nonlocal df

            if dir.parent.name in validation_datasets:
                tc_temp, pyro_temp, temp_regime = getCalData(d, False, False, False, -1)
            else:
                tc_temp, pyro_temp, temp_regime = getCalData(d, False)

            experiment_name = pd.Series(temp_regime, index=range(len(pyro_temp)))
            df_additions = pd.DataFrame({'pyro_temp(°C)':pyro_temp, 'tc_temp(°C)':tc_temp, 'experiment':experiment_name})

            df = pd.concat([df, df_additions], ignore_index=True)

        dataSearch(dir, getDataSubset, id='deg', id_atFront=False)
        df.to_csv(dir / 'Combined_Calibration_Data.csv', index=False)
    else:
        df = pd.read_csv(dir / 'Combined_Calibration_Data.csv')

    data = df.loc[df['experiment'].isin(inclusions)]
    data.reset_index(inplace=True)

    return (data['pyro_temp(°C)'], data['tc_temp(°C)'], data['experiment'])



def regress(data, dir,  batch_iterations=1000, total_iterations=150000, run_directory=None, flag_multiprocessing=False, **kwargs):

    # allow arbitrary modification of operators via kwargs
    binary_operators = ['+', '-', '*', '/', '^']
    unary_operators = ['log', 'neg', 'sqrt', 'exp']

    if 'binary_operators' in kwargs: binary_operators = kwargs['binary_operators']
    if 'unary_operators'  in kwargs: unary_operators  = kwargs['unary_operators'] 

    # unpack data
    pyro_temp, tc_temp, temp_regime = data

    # data must be formatted as pd.Dataframe for regression
    pyro_temp = pd.DataFrame({'Pyrometer_Temperature':pyro_temp})
    tc_temp = pd.DataFrame({'Thermocouple_Temperature':tc_temp})

    experimental_optimization = True

    regressor_args = { 'niterations': batch_iterations, 'batching': True, 'maxsize': 30, \
            'run_id': 'live', 'parallelism': 'multithreading', 'warm_start': True, \
            'bumper': experimental_optimization, 'turbo': experimental_optimization, \
            'model_selection': 'best', 'annealing': True, 'weight_optimize': 0.001, \
            'warmup_maxsize_by':0.1, 'parsimony': 0.05, 'populations': 60, \
            'adaptive_parsimony_scaling': 1500, 'maxdepth': 10, 'denoise': False}
    
    if flag_multiprocessing:
        regressor_args['parallelism'] = 'multiprocessing'
        regressor_args['procs']       = 20
        regressor_args['populations'] = 50
        
    if '^' in binary_operators:
        regressor_args['constraints'] = { '^': (-1, 1)}


    # if we passed an existing run history as an arg, then load in this run history.
    if run_directory is not None:
        model = pysr.PySRRegressor()
        model = model.from_file(run_directory=run_directory, **regressor_args) # warm_start allows us to pick up regression from where we left off

        # read the number of iterations that have been performed on this regressor object previously.
        with open(Path(run_directory) / 'iterations.yaml', 'r') as f:
            
            metadata = yaml.unsafe_load(f)

    # if no run directory is passed, start a new regressor
    else:
        
        model = pysr.PySRRegressor(output_directory=(Path(dir) / 'fits'), binary_operators=binary_operators, \
                                   unary_operators=unary_operators, **regressor_args)
        
        metadata = {}
        metadata['current_iterations'] = 0
        metadata['elapsed_time']       = 0
        metadata['equation_hist']       = []
        metadata['loss_hist']          = []
        
    while metadata['current_iterations'] < total_iterations:
    
        print('Current iterations: ', metadata['current_iterations'])
        print('Elapsed time:       ', timedelta(seconds=metadata['elapsed_time']))
        
        start = time.time()
        m = model.fit(pyro_temp, tc_temp)
        
        metadata['current_iterations'] += model.niterations
        metadata['elapsed_time']       += time.time() - start
        
        metadata['equation_hist'].append(model.get_best()['sympy_format'])
        metadata['loss_hist']    .append(float(model.get_best()['loss']))
        
        with open(Path(model.output_directory) / 'live' / 'iterations.yaml', 'w') as f:
            yaml.dump(metadata, f)
            

    return model

if __name__ == '__main__':
    
    if len(sys.argv) == 2:
        dir = sys.argv[1]
    else:
        dir = selectFolder()
    
    its = 100000

    calibration_datasets = ['60deg', '90deg', '120deg', '130deg', '150deg', '180deg', '205deg', '210deg', 'ambdeg']
    
    validation_data = []

    
    data = combineData(dir, calibration_datasets, force_update=False)
    
    #fit = regress(data, dir, total_iterations=its, run_directory=r"D:\grad data\pyrometer cal no flir\fits\live") #multithread

    fit = regress(data, dir, total_iterations=its, run_directory=r"D:\grad data\new pyro cal\fits\live")


    # read calibration curves from saved files
   #high_fit = pysr.PySRRegressor()
    #fit  = pysr.PySRRegressor()
    
    #fit = fit.from_file(run_directory=r"D:\grad data\new pyro cal\fits\live", model_selection='best')
    #high_fit = high_fit.from_file(run_directory=r"D:\grad data\new_flir\fits\High\20260622_092806_Qs9G0b", model_selection='best')

    """
    vali_data = []
    for data in validation_data:
        validationHigh, validationLow = combineData(dir, [data], True)
        vali_data.append(validationHigh)
        #vali_data.append(validationLow)

    #print(vali_data)
    """


    #plotCalCurve((highRegimeData, lowRegimeData), (high_fit, low_fit), vali_data)
        
    plotCalCurve(data, fit)
    #plotCalCurve((highRegimeData, lowRegimeData), (high_fit, low_fit))

    '''
    high_fit = pysr.PySRRegressor()
    low_fit  = pysr.PySRRegressor()

    high_fit = high_fit.from_file(run_directory=dir + '/fits/High/20260310_175733_l4OvqD', model_selection='best')
    low_fit = low_fit.from_file(run_directory=dir + '/fits/Low/20260311_013003_hNviyE', model_selection='best')

    print(high_fit.sympy(),'\n')
    print(high_fit.equations_)
    print('\n\n-----------------------------------\n\n')
    print(low_fit.sympy(), '\n')
    print(low_fit.equations_)
    '''

    '''
    # need to do this for every validation dataset
    vali_data = getCalData(dir + '/300C High 3/' + 'data_collection_20260303_140755', False, False, False, -1, True)

    validate_response(vali_data, 300)
    '''
    
    


