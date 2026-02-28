import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import subprocess
import sys
import os
import pysr
import sympy

from thermography import getPixels, getFrameData, getHotFrame
from data_manipulation import selectFolder, dfToCsv
from batch_process import dataSearch
from align_data import alignData

# at which frame in the video can we start measuring useful temperature? (discovered manually, accounts for toaster oven door opening)
start_frames = {'60C High':'150', '60C Low':'140', '90C High':'130', '90C Low':'150', '120C High':'140', '120C Low':'190', '150C High':'145', \
                '150C Low':'130', '180C High':'160', '180C Low':'200', '215C High':'180', '230C High':'220', '230C Low':'140', \
                'Ambient High':'200', 'Ambient Low':'200'}

# highlights each pixel in pix in a given frame, used for validation of pixel region selection
def highlight_rect(fr, pix):
    fr = np.float32(fr)/ np.max(fr)

    for row in pix:
        for p in row:
            fr[p[1]][p[0]] = 1.15   # highlights will always be brighter than other frames, even if sensor is saturated

    return fr

# read FLIR temp data from selected pixels & frames, add this thermal data to aligned data csv. returns + saves windowed dataframe
def getCalData(dir, validate_pixel=True, reselect_zone=False, recalc_temps=False, window_length=1):
    # if data isn't stored, calculate it

    if type(dir) is not str:
        dir = dir.path

    temp_regime = os.path.split(os.path.split(dir)[0])[1]
    if not os.access(os.path.split(dir)[0] + '/' + temp_regime + '.csv', os.R_OK) or recalc_temps:
        top_level_dir = os.path.split(os.path.split(dir)[0])[0]

        # data from 021026 and 021126 used different FLIR positions. Because we examine the same set of pixels for each run, we need
        # to store two pixel regions and switch between them based on which measurement we are examining.
        if temp_regime == '230C High' or temp_regime == '230C Low' or temp_regime == 'Ambient High' or temp_regime == 'Ambient Low':
            which_pix = '021126'
        else:
            which_pix = '021026'

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
        avg_temps = []
        for t in range(len(timestamps)):
            sum = 0.0
            for i in range(len(temps[t])):
                for j in range(len(temps[t][i])):
                    sum += temps[t][i][j]
            
            sum /= len(temps[0])*len(temps[0][0])
            avg_temps.append(sum)

        # associate temps with frame names
        cal_data = {}
        for i, fr in enumerate(frames): cal_data[fr] = avg_temps[i]

        df = pd.read_csv(dir + '/aligned_data.csv')

        # add pixel temps to aligned dataset
        df['FLIR_intensity'] = df['FLIR_frame'].map(cal_data)

        # we need to window data in time to cut out obfuscation of wall in video, thermal dropoff towards end of data collection
        first_frame = df['FLIR_frame'][0][:-5] + start_frames[temp_regime] + '.npy'
        start_index = (df['FLIR_frame'] == first_frame).idxmax()

        df = df.loc[(df['time'] > df['time'][start_index]) & (df['time'] < df['time'][start_index] + window_length)]
        df.reset_index(inplace=True)

        # save windowed data to new .csv for later manipulation
        dfToCsv(df, os.path.split(dir)[0] + '/' + temp_regime + '.csv')

    # if windowed data is already saved, just load it from the file
    else:
        df = pd.read_csv(os.path.split(dir)[0] + '/' + temp_regime + '.csv')
    return (df['FLIR_intensity'], df['Channel_0(°C)'], temp_regime)  

# requires input arrays nested in tuples, allows for multiple datasets at once
def plotCalCurve(data, fit=None):

    plt.figure(layout='constrained')
    for i, series in enumerate(data):
        flir_intensity, tc_temp, temp_regime = series

        print(temp_regime.iloc[0])

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

            fit_vals = series_fit.predict(fit_intensity.reshape(-1, 1))
            fit_vals -= 273.15  # convert back to °C

            if len(data) == 1:
                plt.plot(fit_intensity, fit_vals, color='orange', label=('Calibration Curve: ' + str(series_fit.sympy())))
            elif (str(temp_regime.iloc[0])[-4:]).strip() == 'High':
                plt.plot(fit_intensity, fit_vals, color='orange', label=(str('Calibration Curve ' + (str(temp_regime.iloc[0])[-4:]).strip()) + ': ' + str(series_fit.sympy())))
            else:
                plt.plot(fit_intensity, fit_vals, color='purple', label=(str('Calibration Curve ' + (str(temp_regime.iloc[0])[-4:]).strip()) + ': ' + str(series_fit.sympy())))
    
    if len(data) > 1 or fit is not None:
        plt.legend()
    plt.xlabel('FLIR Raw Intensity Value [0:(2^16)-1]')
    plt.ylabel('Thermocouple Temperature (°C)')

    plt.ylim(-150, 1150)
    plt.show()

    return

def combineData(dir, exclusions=[], force_update=False):

    if not os.access(dir + '/Combined_Calibration_Data.csv', os.R_OK) or force_update:
        df = pd.DataFrame({'FLIR_intensity':pd.Series(dtype='float64'), 'tc_temp(°K)':pd.Series(dtype='float64'), 'experiment':pd.Series(dtype='str')})

        def getDataSubset(d):
            nonlocal df
            flir_intensity, tc_temp, temp_regime = getCalData(d, False, False, True)
            tc_temp = tc_temp + 273.15 # convert to kelvin

            experiment_name = pd.Series(temp_regime, index=range(len(flir_intensity)))
            df_additions = pd.DataFrame({'FLIR_intensity':flir_intensity, 'tc_temp(°K)':tc_temp, 'experiment':experiment_name})

            df = pd.concat([df, df_additions], ignore_index=True)

        dataSearch(dir, getDataSubset)
        dfToCsv(df, dir + '/Combined_Calibration_Data.csv')
    else:
        df = pd.read_csv(dir + '/Combined_Calibration_Data.csv')

    high_data = df.loc[df['experiment'].str.endswith('High') & ~df['experiment'].isin(exclusions)]
    low_data = df.loc[df['experiment'].str.endswith('Low') & ~df['experiment'].isin(exclusions)]

    return (high_data['FLIR_intensity'], high_data['tc_temp(°K)'], high_data['experiment']), ((low_data['FLIR_intensity'], low_data['tc_temp(°K)'], low_data['experiment']))

def symb_regression(x, y, niterations):

    def custom_loss(y_pred, y_true):
        mse = np.mean((y_pred - y_true) ** 2)


    #model = pysr.PySRRegressor(binary_operators=['+', '-', '*', '/', '^'], unary_operators=['inv', 'exp', 'log'], \
    #                           niterations=niterations, batching=True, maxsize=30, constraints={'^':(-1, 1)})

    model = pysr.PySRRegressor(binary_operators=['+', '*'], unary_operators=['log'], \
                            niterations=niterations, batching=True, maxsize=30)
    
    model.fit(x,y)

    print(model.sympy())
    #do something with model
    return model

def regress(data, niterations=1000):
    flir_intensity, tc_temp, temp_regime = data

    flir_intensity = pd.DataFrame({'FLIR_Intensity':flir_intensity})
    tc_temp = pd.DataFrame({'Thermocouple_Temperature(°K)':tc_temp})

    fit = symb_regression(flir_intensity, tc_temp, niterations)

    return fit

# BIG ISSUE: running getCalData with recalc_temps=True more than once is somehow changing FLIR_frame to break the mapping of cal_data
if __name__ == '__main__':
    dir = selectFolder()

    its = 20000

    highRegimeData, lowRegimeData = combineData(dir, ['230C Low'])
    high_fit = regress(highRegimeData, its)
    low_fit = regress(lowRegimeData, its)

    plotCalCurve((highRegimeData, lowRegimeData), (high_fit, low_fit))


