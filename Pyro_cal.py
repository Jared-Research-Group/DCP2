import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'build', 'lib.win-amd64-cpython-310')
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)
    
import pysr
import sympy
from sympy.abc import x
import math
from datetime import datetime, timedelta

from thermography import getPixels, getFrameData, getHotFrame
from data_manipulation import selectFolder, dfToCsv
from batch_process import dataSearch
from align_data import alignData

# at which frame in the video can we start measuring useful temperature? (discovered manually, accounts for toaster oven door opening)
#start_frames = {'60C High':'150', '60C Low':'140', '90C High':'130', '90C Low':'150', '120C High':'140', '120C Low':'190', '150C High':'145', \
#                '150C Low':'130', '180C High':'160', '180C Low':'200', '215C High':'180', '230C High':'220', '230C Low':'140', \
#                'Ambient High':'200', 'Ambient Low':'200', '250C High':'0', '300C High 1':'0', '300C High 2':'0', '300C High 3':'0', \
#                'Cold High':'0', 'Cold Low':'0', 'dark high':'0', 'dark low':'0', 'Warming High':'0', '400C High':'0', 'burning_kaptan High':'0', '500C High':'0'}

#positions = {'60C High':'021026', '60C Low':'021026', '90C High':'021026', '90C Low':'021026', '120C High':'021026', '120C Low':'021026', '150C High':'021026', \
#                '150C Low':'021026', '180C High':'021026', '180C Low':'021026', '215C High':'021026', '230C High':'021126', '230C Low':'021126', \
#                'Ambient High':'021126', 'Ambient Low':'021126', 'Cold High':'022726', 'Cold Low':'022726', 'Warming High':'022726', '250C High':'030326', \
#                '300C High 1':'030326', '300C High 2':'030326', '300C High 3':'030326', 'dark low':'022726', 'dark high':'022726', '400C High':'031226_2', 'burning_kaptan High':'031226_1', '500C High':'031326'}

#validation_datasets = ['250C High', '300C High 1', '300C High 2', '300C High 3', 'Warming High', '400C High', 'burning_kaptan High', '500C High']

# highlights each pixel in pix in a given frame, used for validation of pixel region selection
def highlight_rect(fr, pix):
    fr = np.float32(fr)/ np.max(fr)

    for row in pix:
        for p in row:
            fr[p[1]][p[0]] = 1.15   # highlights will always be brighter than other frames, even if sensor is saturated

    return fr

# read FLIR temp data from selected pixels & frames, add this thermal data to aligned data csv. returns + saves windowed dataframe
def getCalData(dir, validate_pixel=True, reselect_zone=False, recalc_temps=False, window_length=1, needTimes=False):
    # if data isn't stored, calculate it

    if type(dir) is not str:
        dir = dir.path

    temp_regime = os.path.split(os.path.split(dir)[0])[1]
    if (window_length != -1 and not os.access(os.path.split(dir)[0] + '/' + temp_regime + '.csv', os.R_OK)) or (window_length == -1 and not os.access(os.path.split(dir)[0] + '/' + temp_regime + '_unwindowed.csv', os.R_OK)) or recalc_temps:
        top_level_dir = os.path.split(os.path.split(dir)[0])[0]

        # data from 021026, 021126, 022726, and 030326 used different FLIR positions. Because we examine the same set of pixels for each run, we need
        # to store two pixel regions and switch between them based on which measurement we are examining.
        which_pix = 'pyr'

        # if we can't find stored pixel position, request it from user
        if not os.access(top_level_dir + '/pix' + which_pix + '.npy', os.R_OK) or reselect_zone:
            p = getPixels(dir + '/FLIR', 2)
            p = np.array(p)
            np.save(top_level_dir + '/pix' + which_pix + '.npy', p)
        else:
            p = np.load(top_level_dir + '/pix' + which_pix + '.npy')

        # display selected pixels to validate position of zone
        if validate_pixel:
            import matplotlib.pyplot as plt

            ex_frame = getHotFrame(dir + '/FLIR')
            ex_frame = highlight_rect(ex_frame, p)
            plt.imshow(ex_frame, cmap='viridis')
            plt.grid(False)
            plt.show()

        # collect data on selected pixels
        timestamps, temps, frames = getFrameData(dir + '/FLIR', p)

        # average temps of pixels in selected zone for each frame
        cal_data = {}
        for t, fr in enumerate(frames):
            sum = 0.0
            for i in range(len(temps[t])):
                for j in range(len(temps[t][i])):
                    sum += temps[t][i][j]
            
            sum /= len(temps[0])*len(temps[0][0])

            cal_data[os.path.abspath(fr)] = sum

        df = pd.read_csv(dir + '/aligned_data.csv')

        # add pixel temps to aligned dataset
        for i, fr in enumerate(df['FLIR_frame']):
            df.loc[i, 'FLIR_frame'] = os.path.abspath(fr)

        df['FLIR_intensity'] = df['FLIR_frame'].map(cal_data)

        # we need to window data in time to cut out obfuscation of wall in video, thermal dropoff towards end of data collection
        if window_length != -1:
            first_frame = os.path.split(df['FLIR_frame'][0])[0] + '\\FLIR-Frame-' + 'pyr.npy'
            print(first_frame)

            start_index = (df['FLIR_frame'] == first_frame).idxmax()
            print(start_index)

            print(type(df['time'][0]))
            df = df.loc[(df['time'] > df['time'][start_index]) & (df['time'] < (df['time'][start_index] + window_length))]
            df.reset_index(inplace=True)

            # save windowed data to new .csv for later manipulation
            dfToCsv(df, os.path.split(dir)[0] + '/' + temp_regime + '.csv')

        else:
            dfToCsv(df, os.path.split(dir)[0] + '/' + temp_regime + '_unwindowed.csv')

    # if windowed data is already saved, just load it from the file
    else:
        if window_length != -1:
            df = pd.read_csv(os.path.split(dir)[0] + '/' + temp_regime + '.csv')
        else:
            df = pd.read_csv(os.path.split(dir)[0] + '/' + temp_regime + '_unwindowed.csv')

    channel_num = 0

    #if 'kaptan' in temp_regime or '400' in temp_regime:
    #    channel_num = 1

    if needTimes:
        return (df['FLIR_intensity'], df['Channel_' + str(channel_num) + '(°C)'], df['Pyrometer_Temp(°C)'], temp_regime, df['time'])
    else:
        return (df['FLIR_intensity'], df['Channel_' + str(channel_num) + '(°C)'], df['Pyrometer_Temp(°C)'], temp_regime)  

# requires input arrays nested in tuples, allows for multiple datasets at once
def plotCalCurve(data, fit=None, vali_data=None):

    plt.figure(layout='constrained')
    for i, series in enumerate(data):
        flir_intensity, tc_temp, temp_regime = series

        tc_temp -= 273.15        # convert back to °C

        if len(data) == 1:
            plt.scatter(flir_intensity, tc_temp, color='blue', label='Measured Values')
        elif (str(temp_regime.iloc[0])[-4:]).strip() == 'High':
            plt.scatter(flir_intensity, tc_temp, color='blue', label=str('Measured Values ' + (str(temp_regime.iloc[0])[-4:]).strip()))
        else:
            plt.scatter(flir_intensity, tc_temp, color='green', label=str('Measured Values ' + (str(temp_regime.iloc[0])[-4:]).strip()))

        if fit is not None:
            series_fit = fit[i]

            rng = 2**16 - 1
            fit_intensity = np.linspace(0, 2**16-1, 10000)

            if type(series_fit) == pysr.PySRRegressor:
                fit_vals = series_fit.predict(fit_intensity.reshape(-1, 1))
                fit_str = str(series_fit.sympy())
            else:
                fit_str = str(series_fit)
                series_fit = sympy.lambdify(x, series_fit, "numpy")
                fit_vals = series_fit(fit_intensity)


            fit_vals -= 273.15  # convert back to °C

            if len(data) == 1:
                plt.plot(fit_intensity, fit_vals, color='orange', label=('Calibration Curve: ' + fit_str))
            elif (str(temp_regime.iloc[0])[-4:]).strip() == 'High':
                plt.plot(fit_intensity, fit_vals, color='orange', label=(str('Calibration Curve ' + (str(temp_regime.iloc[0])[-4:]).strip()) + ': ' + fit_str))
            else:
                plt.plot(fit_intensity, fit_vals, color='purple', label=(str('Calibration Curve ' + (str(temp_regime.iloc[0])[-4:]).strip()) + ': ' + fit_str))

    if vali_data is not None:
        for i, data in enumerate(vali_data):
            vali_intensity, vali_temp, vali_name = data
            vali_temp -= 273.15

            plt.scatter(vali_intensity, vali_temp, label='Validation Curve ' + str(i), s=0.05)

    
    if len(data) > 1 or fit is not None or vali_data is not None:
        plt.legend()
    plt.xlabel('FLIR Raw Intensity Value [0:(2^16)-1]')
    plt.ylabel('Thermocouple Temperature (°C)')

    plt.ylim(-150, 1150)
    plt.show()

    return

def combineData(dir, inclusions, validation=False, force_update=False):

    if not os.access(dir + '/Combined_Calibration_Data.csv', os.R_OK) or force_update:
        df = pd.DataFrame({'FLIR_intensity':pd.Series(dtype='float64'), 'tc_temp(°K)':pd.Series(dtype='float64'), 'pyr_temp(°K)':pd.Series(dtype='float64'), 'experiment':pd.Series(dtype='str')})

        def getDataSubset(d):
            nonlocal df

            #if os.path.split(os.path.split(d)[0])[1] in validation_datasets:
            #    flir_intensity, tc_temp, temp_regime = getCalData(d, False, False, False, -1)
            #else:
                #flir_intensity, tc_temp, temp_regime = getCalData(d, False)
            flir_intensity, tc_temp, pyr_temp, temp_regime = getCalData(d, False)
            tc_temp = tc_temp + 273.15 # convert to kelvin
            pyr_temp = pyr_temp + 273.15

            experiment_name = pd.Series(temp_regime, index=range(len(flir_intensity)))
            df_additions = pd.DataFrame({'FLIR_intensity':flir_intensity, 'tc_temp(°K)':tc_temp, 'pyr_temp(°K)':pyr_temp, 'experiment':experiment_name})

            df = pd.concat([df, df_additions], ignore_index=True)

        dataSearch(dir, getDataSubset)
        dfToCsv(df, dir + '/Combined_Calibration_Data.csv')
    else:
        df = pd.read_csv(dir + '/Combined_Calibration_Data.csv')

    high_data = df.loc[df['experiment'].str.contains('High') & df['experiment'].isin(inclusions)]
    high_data.reset_index(inplace=True)

    low_data = df.loc[df['experiment'].str.contains('Low') & df['experiment'].isin(inclusions)]
    low_data.reset_index(inplace=True)

    return (high_data['FLIR_intensity'], high_data['tc_temp(°K)'], high_data['pyr_temp(°K)'], high_data['experiment']), ((low_data['FLIR_intensity'], low_data['tc_temp(°K)'], low_data['pyr_temp(°K)'], low_data['experiment']))


def regress(data, d,  niterations=1000):
    flir_intensity, tc_temp, pyr_temp, temp_regime = data

    flir_intensity = pd.DataFrame({'FLIR_Intensity':flir_intensity})
    tc_temp = pd.DataFrame({'Thermocouple_Temperature(°K)':tc_temp})
    pyr_temp = pd.DataFrame({'Pyrometer Temperature(°K)':pyr_temp})

    model_type = ''
    if 'High' in temp_regime[0]:
        model_type = 'High'
    elif 'Low' in temp_regime[0]:
        model_type = 'Low'

    #model = pysr.PySRRegressor(binary_operators=['+', '*', '^'], unary_operators=['log', 'neg', 'exp'], \
    #                        niterations=niterations, batching=True, maxsize=30, output_directory=(d + '/fits/' + model_type + '/'), \
    #                        constraints={'^':(-1,2)})

    model = pysr.PySRRegressor(binary_operators=['+', '*'], unary_operators=['log', 'neg'], \
                            niterations=niterations, batching=True, maxsize=30, output_directory=(d + '/fits/' + model_type + '/'))
    
    model.fit(flir_intensity, tc_temp)

    return model

# currently doing a weird reversed semi-log to validate 120C High
def validate_response(vali_data, window=300):
    vali_intensity, vali_temp, temp_regime, vali_time = vali_data

    #vali_intensity = vali_intensity[::-1]
    #vali_temp = vali_temp[::-1]

    fig, ax1 = plt.subplots(layout='constrained')
    ax1.scatter(vali_time[window:-window], vali_temp[window:-window], label='TC Curve', alpha=0.5, c='red')
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('Value Normalized to Thermocouple Temperature (°C)')
    ax1.tick_params(axis='y', labelcolor='red')

    #ax1.set_xscale('log')
   
    ax1.scatter(vali_time[window:-window], (vali_intensity[window:-window] * max(vali_temp[window:-window]))/max(vali_intensity[window:-window]), label='FLIR Intensity', alpha=0.5, c='blue')

    fig.legend()
    plt.show()

    def fd(y, x, i, step=10):
        return (y[i+step] - y[i])/(x[i+step] - x[i])

    d_intensity = []
    d_temp = []
    step = 25
    for i, t in enumerate(vali_intensity[:-step]):
        d_intensity.append(fd(vali_intensity, vali_time, i, step))
        d_temp.append(fd(vali_temp, vali_time, i, step))

    d_temp = np.array(d_temp)
    d_intensity = np.array(d_intensity)

    fig, ax1 = plt.subplots(layout='constrained')
    ax1.scatter(vali_time[window:-window - step], d_temp[window:-window], label='TC Curve', alpha=0.5, s=5, c='red')
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('Value Normalized to Thermocouple Heating Rate (°C/s)')
    ax1.tick_params(axis='y', labelcolor='red')

    ax1.scatter(vali_time[window:-window - step], (d_intensity[window:-window] * max(abs(d_temp[window:-window])))/max(abs(d_intensity[window:-window])), label='FLIR Intensity', alpha=0.5, s=5, c='blue')

    fig.legend()
    plt.show()

    return

if __name__ == '__main__':
    
    dir = selectFolder()
    
    its = 100000

    calibration_datasets = ['data_collection_20260331_145851', 'data_collection_20260331_150208', 'data_collection_20260331_150516', 'data_collection_20260331_150916', 'data_collection_20260331_151325',
                            'data_collection_20260331_151701', 'data_collection_20260331_152125', 'data_collection_20260331_152423', 'data_collection_20260331_152825', 'data_collection_20260331_153059', 
                            'data_collection_20260331_153347', 'data_collection_20260331_153853', 'data_collection_20260331_154356', 'data_collection_20260331_154938', 'data_collection_20260331_155504',
                            ]
    #validation_data = ['250C High', '300C High 1', '300C High 2', '300C High 3', 'Warming High', 'burning_kaptan High', '500C High']
    #validation_data = ['230C High']

    
    highRegimeData, lowRegimeData = combineData(dir, calibration_datasets)
    
    print(highRegimeData)
    print(lowRegimeData)

    fit = regress(highRegimeData, dir, its)


    #high_fit = high_fit.from_file(run_directory=os.getcwd() + '/FLIR_fits/High', model_selection='best')
    #low_fit = low_fit.from_file(run_directory=os.getcwd() + '/FLIR_fits/Low', model_selection='best')

    '''
    vali_data = []
    for data in validation_data:
        validationHigh, validationLow = combineData(dir, [data], True)
        vali_data.append(validationHigh)
        #vali_data.append(validationLow)

    '''

    #print(vali_data)


    plotCalCurve((highRegimeData, lowRegimeData), (high_fit, low_fit))

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
    
    


