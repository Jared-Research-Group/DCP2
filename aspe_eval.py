import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

import random

build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'build', 'lib.win-amd64-cpython-310')
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)
    
import pysr
import sympy
from sympy.abc import x
from datetime import timedelta
import logging

from thermography import getPixels, getFrameData, getHotFrame
from batch_process import dataSearch
from pyrometer import getPyrometerData

import helper_functions

# highlights each pixel in pix in a given frame, used for validation of pixel region selection
def highlight_rect(fr, pix):
    fr = np.float32(fr)/ np.max(fr)

    for row in pix:
        for p in row:
            fr[p[1]][p[0]] = 1.15   # highlights will always be brighter than other frames, even if sensor is saturated

    return fr

# read FLIR temp data from selected pixels & frames, add this thermal data to aligned data csv. returns + saves windowed dataframe
def getFlirData(dir, validate_pixel=True, reselect_zone=False, recalc_temps=False, window_length=20):
    # if data isn't stored, calculate it

    dir = Path(dir)

    run_name = dir.parents[3].name + '_' + dir.parents[2].name + '_' + dir.parents[1].name + '_' + dir.parents[0].name
    if  not os.access(dir.parent / (run_name + '.csv'), os.R_OK) or recalc_temps:
        top_level_dir = dir.parents[3]
        which_pix = run_name
        
    
        if dir.parents[1].name == 'ambient':
            #return (pd.Series(), pd.Series(), run_name)  
            which_pix = dir.parents[3].name + '_' + dir.parents[2].name + '_hot_' + dir.parents[0].name

        print(top_level_dir / ('pix' + which_pix + '.npy'))

        # if we can't find stored pixel position, request it from user
        if not os.access(top_level_dir / ('pix' + which_pix + '.npy'), os.R_OK) or reselect_zone:
            
            print(which_pix)
            
            p = getPixels(dir / 'FLIR', 2)
            p = np.array(p)

            np.save(top_level_dir / ('pix' + which_pix + '.npy'), p)
        else:
            p = np.load(top_level_dir / ('pix' + which_pix + '.npy'))

        # display selected pixels to validate position of zone
        if validate_pixel:
            import matplotlib.pyplot as plt

            ex_frame = getHotFrame(dir / 'FLIR')
            ex_frame = highlight_rect(ex_frame, p)
            plt.imshow(ex_frame, cmap='viridis')
            plt.grid(False)
            plt.show()

        # collect data on selected pixels
        timestamps, temps, frames = getFrameData(dir / 'FLIR', p)

        # average temps of pixels in selected zone for each frame
        cal_data = {}
        for t, fr in enumerate(frames):
            sum = 0.0
            for i in range(len(temps[t])):
                for j in range(len(temps[t][i])):
                    sum += temps[t][i][j]
            
            sum /= len(temps[0])*len(temps[0][0])

            cal_data[os.path.abspath(fr)] = sum

        df = pd.read_csv(dir / 'aligned_data.csv')

        # add pixel temps to aligned dataset
        for i, fr in enumerate(df['FLIR_frame']):
            df.loc[i, 'FLIR_frame'] = os.path.abspath(fr)

        df['FLIR_intensity'] = df['FLIR_frame'].map(cal_data)

        # we need to window data in time to cut out obfuscation of wall in video, thermal dropoff towards end of data collection
        if window_length != -1:
            first_frame = Path(df['FLIR_frame'][0]).parent / 'FLIR-Frame-50.npy'

            start_index = (df['FLIR_frame'] == first_frame).idxmax()

            df = df.loc[(df['time'] > df['time'][start_index]) & (df['time'] < (df['time'][start_index] + window_length))]
            df.reset_index(inplace=True)

            # save windowed data to new .csv for later manipulation
            df.to_csv(dir.parent / (run_name + '.csv'))

    # if windowed data is already saved, just load it from the file
    else:
        df = pd.read_csv(dir.parent / (run_name + '.csv'))

    if dir.parents[2].name == 'clean':
        channel_num = 0
    else:
        channel_num = 1

    return (df['FLIR_intensity'], df['Channel_' + str(channel_num) + '(°C)'], run_name)  

# read FLIR temp data from selected pixels & frames, add this thermal data to aligned data csv. returns + saves windowed dataframe
def getPyroData(dir, recalc_temps=False, window_length=20):
    # if data isn't stored, calculate it

    dir = Path(dir)

    run_name = dir.parents[2].name + '_' + dir.parents[1].name + '_' + dir.parents[0].name + '_' + dir.name
    if  not os.access(dir.parent / (run_name + '.csv'), os.R_OK) or recalc_temps:
        top_level_dir = dir.parents[3]

        df = pd.read_csv(dir / 'aligned_data.csv')

        # we need to window data in time to cut out obfuscation of wall in video, thermal dropoff towards end of data collection
        if window_length != -1:

            start_index = 10

            df = df.loc[(df['time'] > df['time'][start_index]) & (df['time'] < (df['time'][start_index] + window_length))]
            df.reset_index(inplace=True)

            # save windowed data to new .csv for later manipulation
            df.to_csv(dir.parent / (run_name + '.csv'))

    # if windowed data is already saved, just load it from the file
    else:
        df = pd.read_csv(dir.parent / (run_name + '.csv'))

    if dir.parents[1].name == 'clean':
        channel_num = 0
    else:
        channel_num = 1

    return (df['Pyrometer_Temp(°C)'], df['Channel_' + str(channel_num) + '(°C)'], run_name)  

def combineFlirData(dir, force_update=False):
    
    dir = str(dir)

    if not os.access(dir + '/Combined_FLIR_Data.csv', os.R_OK) or not os.access(dir + '/Combined_Pyro_Data.csv', os.R_OK) or force_update:
        df = pd.DataFrame({'FLIR_intensity':pd.Series(dtype='float64'), 'tc_temp(°C)':pd.Series(dtype='float64'), 'experiment':pd.Series(dtype='str')})

        def getDataSubset(d):
            nonlocal df

            flir_intensity, tc_temp, run_name = getFlirData(d, reselect_zone=False, validate_pixel=False, recalc_temps=force_update)
            
            experiment_name = pd.Series(run_name, index=range(len(flir_intensity)))
            df_additions = pd.DataFrame({'FLIR_intensity':flir_intensity, 'tc_temp(°C)':tc_temp, 'experiment':experiment_name})
            df = pd.concat([df, df_additions], ignore_index=True)

        dataSearch(dir, getDataSubset)
        
        
        # get FLIR models
        x = sympy.symbols('FLIR_Intensity')

        logging.getLogger('pysr').setLevel(logging.WARNING)

        high_fit   = pysr.PySRRegressor().from_file(run_directory=r"D:\grad data\new_flir\fits\GREAT BUT 10s\live", model_selection='best', verbosity=0)
        high_model = sympy.lambdify(x, high_fit.sympy(11), modules='numpy')
            
        df['FLIR_temp(°C)'] = helper_functions.flirConversion(df['FLIR_intensity'], high_model)
        
        df.to_csv( dir + '/Combined_FLIR_Data.csv')
        
    else:
        df = pd.read_csv(dir + '/Combined_FLIR_Data.csv')

    df.reset_index(inplace=True)

    # break data into clean and sooty, ambient and hot
    clean_wall_data = df[df['experiment'].str.contains('clean')]
    sooty_wall_data = df[df['experiment'].str.contains('sooty')]
    
    clean_amb = clean_wall_data[clean_wall_data['experiment'].str.contains('ambient')]
    clean_hot = clean_wall_data[clean_wall_data['experiment'].str.contains('hot')]
    sooty_amb = sooty_wall_data[sooty_wall_data['experiment'].str.contains('ambient')]
    sooty_hot = sooty_wall_data[sooty_wall_data['experiment'].str.contains('hot')]
    
    angles = np.arange(5) * 15 + 30
    
    clean_flir_ambient_temp = []
    clean_flir_hot_temp     = []
    clean_tc_ambient_temp   = []
    clean_tc_hot_temp       = []
    sooty_flir_ambient_temp = []
    sooty_flir_hot_temp     = []
    sooty_tc_ambient_temp   = []
    sooty_tc_hot_temp       = []
    
    for i, id in enumerate(np.unique(angles)):
        id = str(id)
        
        clean_flir_ambient_temp.append(clean_amb[clean_amb['experiment'].str.contains(id)]['FLIR_temp(°C)'])
        clean_tc_ambient_temp.append(clean_amb[clean_amb['experiment'].str.contains(id)]['tc_temp(°C)'])
        clean_flir_hot_temp.append(clean_hot[clean_hot['experiment'].str.contains(id)]['FLIR_temp(°C)'])
        clean_tc_hot_temp.append(clean_hot[clean_hot['experiment'].str.contains(id)]['tc_temp(°C)'])
        sooty_flir_ambient_temp.append(sooty_amb[sooty_amb['experiment'].str.contains(id)]['FLIR_temp(°C)'])
        sooty_tc_ambient_temp.append(sooty_amb[sooty_amb['experiment'].str.contains(id)]['tc_temp(°C)'])
        sooty_flir_hot_temp.append(sooty_hot[sooty_hot['experiment'].str.contains(id)]['FLIR_temp(°C)'])
        sooty_tc_hot_temp.append(sooty_hot[sooty_hot['experiment'].str.contains(id)]['tc_temp(°C)'])
        
    clean_flir_ambient_temp = np.array(clean_flir_ambient_temp)
    clean_tc_ambient_temp   = np.array(clean_tc_ambient_temp)
    clean_flir_hot_temp     = np.array(clean_flir_hot_temp)
    clean_tc_hot_temp       = np.array(clean_tc_hot_temp)
    sooty_flir_ambient_temp = np.array(sooty_flir_ambient_temp)
    sooty_tc_ambient_temp   = np.array(sooty_tc_ambient_temp)
    sooty_flir_hot_temp     = np.array(sooty_flir_hot_temp)
    sooty_tc_hot_temp       = np.array(sooty_tc_hot_temp)
    
    fig, ax = plt.subplots(2, 1, layout='constrained', sharex=True)
    
    fig.suptitle('Ambient Temp. Clean Wall')
    
    ax[0].scatter(angles, np.mean(clean_flir_ambient_temp, axis=1), c='blue', label='FLIR')
    ax[0].scatter(angles, np.mean(clean_tc_ambient_temp, axis=1),   c ='blue', marker='s', alpha=.7, label='Thermocouple')

    ax[0].set_title('Data')
    ax[0].set_xlabel('Viewing Angle (°)')
    ax[0].set_ylabel('Average Temperature (°C)')
    ax[0].set_ylim(20, 24)
    
    ax[1].plot(angles, np.mean((clean_flir_ambient_temp - clean_tc_ambient_temp)**2, axis=1), c='blue', marker='.')
    
    ax[1].set_title('Error')
    ax[1].set_xlabel('Viewing Angle (°)')
    ax[1].set_ylabel('Mean Square Error')
    ax[1].set_ylim(4, 9)
    
    ax[0].legend()
    ax[0].grid(True)
    ax[1].grid(True)
    
    plt.show()
    
    fig, ax = plt.subplots(2, 1, layout='constrained', sharex=True)
    
    fig.suptitle('Elevated Temp. Clean Wall')
    
    ax[0].scatter(angles, np.mean(clean_flir_hot_temp, axis=1), c='red', label='FLIR')
    ax[0].scatter(angles, np.mean(clean_tc_hot_temp, axis=1),   c ='red', marker='s', alpha=.7, label='Thermocouple')

    ax[0].set_title('Data')
    ax[0].set_xlabel('Viewing Angle (°)')
    ax[0].set_ylabel('Average Temperature (°C)')
    ax[0].set_ylim(100, 150)
    
    ax[1].plot(angles, np.mean((clean_flir_hot_temp - clean_tc_hot_temp)**2, axis=1), c='red', marker='.')
    
    ax[1].set_title('Error')
    ax[1].set_xlabel('Viewing Angle (°)')
    ax[1].set_ylabel('Mean Square Error')
    ax[1].set_ylim(30, 275)
    
    ax[0].legend()
    ax[0].grid(True)
    ax[1].grid(True)
    plt.show()
    
    fig, ax = plt.subplots(2, 1, layout='constrained', sharex=True)
    
    fig.suptitle('Ambient Temp. Sooty Wall')
    
    ax[0].scatter(angles, np.mean(sooty_flir_ambient_temp, axis=1), c='green', label='FLIR')
    ax[0].scatter(angles, np.mean(sooty_tc_ambient_temp, axis=1),   c ='green', marker='s', alpha=.7, label='Thermocouple')

    ax[0].set_title('Data')
    ax[0].set_xlabel('Viewing Angle (°)')
    ax[0].set_ylabel('Average Temperature (°C)')
    ax[0].set_ylim(20, 24)
    
    ax[1].plot(angles, np.mean((sooty_flir_ambient_temp - sooty_tc_ambient_temp)**2, axis=1), c='green', marker='.')
    
    ax[1].set_title('Error')
    ax[1].set_xlabel('Viewing Angle (°)')
    ax[1].set_ylabel('Mean Square Error')
    ax[1].set_ylim(4, 9)
    
    ax[0].legend()
    ax[0].grid(True)
    ax[1].grid(True)
    
    plt.show()
    
    fig, ax = plt.subplots(2, 1, layout='constrained', sharex=True)
    
    fig.suptitle('Elevated Temp. Sooty Wall')
    
    ax[0].scatter(angles, np.mean(sooty_flir_hot_temp, axis=1), c='purple', label='FLIR')
    ax[0].scatter(angles, np.mean(sooty_tc_hot_temp, axis=1),   c ='purple', marker='s', alpha=.7, label='Thermocouple')

    ax[0].set_title('Data')
    ax[0].set_xlabel('Viewing Angle (°)')
    ax[0].set_ylabel('Average Temperature (°C)')
    ax[0].set_ylim(100, 150)
    
    ax[1].plot(angles, np.mean((sooty_flir_hot_temp - sooty_tc_hot_temp)**2, axis=1), c='purple', marker='.')
    
    ax[1].set_title('Error')
    ax[1].set_xlabel('Viewing Angle (°)')
    ax[1].set_ylabel('Mean Square error')
    ax[1].set_ylim(30, 275)
    
    ax[0].legend()
    ax[0].grid(True)
    ax[1].grid(True)
    plt.show()
    
    '''
    fig, ax = plt.subplots(1, 1, layout='constrained', sharex=True)
    
    ax.set_title('Combined MSE')
    ax.set_ylabel('MSE')
    ax.set_xlabel('Viewing Angle (°)')
    
    ax.plot(angles, np.mean((clean_flir_ambient_temp - clean_tc_ambient_temp)**2, axis=1), c='blue', marker='.', label='Ambient Clean')
    ax.plot(angles, np.mean((clean_flir_hot_temp - clean_tc_hot_temp)**2, axis=1), c='red', marker='.', label='Hot Clean')
    ax.plot(angles, np.mean((sooty_flir_ambient_temp - sooty_tc_ambient_temp)**2, axis=1), c='green', marker='.', label='Ambient Sooty')
    ax.plot(angles, np.mean((sooty_flir_hot_temp - sooty_tc_hot_temp)**2, axis=1), c='purple', marker='.', label='Hot Sooty')
    
    ax.legend()
    ax.grid(True)
    
    plt.show()
    
    for i in range(5):
        
        fig, ax = plt.subplots(2,2, layout='constrained')
        
        ax[0,0].hist2d(clean_tc_ambient_temp[i], clean_flir_ambient_temp[i])
        ax[0,1].hist2d(clean_tc_hot_temp[i], clean_flir_hot_temp[i])
        ax[1,0].hist2d(sooty_tc_ambient_temp[i], sooty_flir_ambient_temp[i])
        ax[1,1].hist2d(sooty_tc_hot_temp[i], sooty_flir_hot_temp[i])
        
        ax[0,0].set_title('Clean Ambient')
        ax[0,1].set_title('Clean Hot')
        ax[1,0].set_title('Sooty Ambient')
        ax[1,1].set_title('Sooty Hot')
        
        fig.suptitle('Angle: ' + str(angles[i]))
        
        plt.show()
    '''

    return df

def combinePyroData(dir, force_update=False):
    
    dir = str(dir)

    if not os.access(dir + '/Combined_Pyro_Data.csv', os.R_OK) or not os.access(dir + '/Combined_Pyro_Data.csv', os.R_OK) or force_update:
        df = pd.DataFrame({'pyro_temp(°C)':pd.Series(dtype='float64'), 'tc_temp(°C)':pd.Series(dtype='float64'), 'experiment':pd.Series(dtype='str')})

        def getDataSubset(d):
            nonlocal df

            pyro_temp, tc_temp, run_name = getPyroData(d, recalc_temps=force_update)
            experiment_name = pd.Series(run_name, index=range(len(pyro_temp)))
            df_additions = pd.DataFrame({'pyro_temp(°C)':pyro_temp, 'tc_temp(°C)':tc_temp, 'experiment':experiment_name})
            df = pd.concat([df, df_additions], ignore_index=True)

        dataSearch(dir, getDataSubset, id='deg', id_atFront=False)
        
        # get Pyro model
        x = sympy.symbols('Pyrometer_Temperature')

        logging.getLogger('pysr').setLevel(logging.WARNING)

        pyro_fit   = pysr.PySRRegressor().from_file(run_directory=r"D:\grad data\new pyro cal\fits\live", model_selection='best', verbosity=0)
        pyro_model = sympy.lambdify(x, pyro_fit.sympy(7), modules='numpy')
            
        df['pyro_temp_calibrated(°C)'] = pyro_model(df['pyro_temp(°C)'])
        
        
        df.to_csv(dir + '/Combined_Pyro_Data.csv', index=False)
        
    else:
        df = pd.read_csv(dir + '/Combined_Pyro_Data.csv')

    df.reset_index(inplace=True)
    
       # break data into clean and sooty, ambient and hot
    clean_wall_data = df[df['experiment'].str.contains('clean')]
    sooty_wall_data = df[df['experiment'].str.contains('sooty')]
    
    clean_amb = clean_wall_data[clean_wall_data['experiment'].str.contains('ambient')]
    clean_hot = clean_wall_data[clean_wall_data['experiment'].str.contains('hot')]
    sooty_amb = sooty_wall_data[sooty_wall_data['experiment'].str.contains('ambient')]
    sooty_hot = sooty_wall_data[sooty_wall_data['experiment'].str.contains('hot')]
    
    angles = np.arange(5) * 15 + 30
    
    clean_pyro_ambient_temp = []
    clean_pyro_hot_temp     = []
    clean_tc_ambient_temp   = []
    clean_tc_hot_temp       = []
    sooty_pyro_ambient_temp = []
    sooty_pyro_hot_temp     = []
    sooty_tc_ambient_temp   = []
    sooty_tc_hot_temp       = []
    
    for i, id in enumerate(np.unique(angles)):
        id = str(id)
        
        clean_pyro_ambient_temp.append(clean_amb[clean_amb['experiment'].str.contains(id)]['pyro_temp_calibrated(°C)'])
        clean_tc_ambient_temp.append(clean_amb[clean_amb['experiment'].str.contains(id)]['tc_temp(°C)'])
        clean_pyro_hot_temp.append(clean_hot[clean_hot['experiment'].str.contains(id)]['pyro_temp_calibrated(°C)'])
        clean_tc_hot_temp.append(clean_hot[clean_hot['experiment'].str.contains(id)]['tc_temp(°C)'])
        sooty_pyro_ambient_temp.append(sooty_amb[sooty_amb['experiment'].str.contains(id)]['pyro_temp_calibrated(°C)'])
        sooty_tc_ambient_temp.append(sooty_amb[sooty_amb['experiment'].str.contains(id)]['tc_temp(°C)'])
        sooty_pyro_hot_temp.append(sooty_hot[sooty_hot['experiment'].str.contains(id)]['pyro_temp_calibrated(°C)'])
        sooty_tc_hot_temp.append(sooty_hot[sooty_hot['experiment'].str.contains(id)]['tc_temp(°C)'])
        
    clean_pyro_ambient_temp = np.array(clean_pyro_ambient_temp)
    clean_tc_ambient_temp   = np.array(clean_tc_ambient_temp)
    clean_pyro_hot_temp     = np.array(clean_pyro_hot_temp)
    clean_tc_hot_temp       = np.array(clean_tc_hot_temp)
    sooty_pyro_ambient_temp = np.array(sooty_pyro_ambient_temp)
    sooty_tc_ambient_temp   = np.array(sooty_tc_ambient_temp)
    sooty_pyro_hot_temp     = np.array(sooty_pyro_hot_temp)
    sooty_tc_hot_temp       = np.array(sooty_tc_hot_temp)
    
    fig, ax = plt.subplots(2, 1, layout='constrained', sharex=True)
    
    fig.suptitle('Ambient Temp. Clean Wall')
    
    ax[0].scatter(angles, np.mean(clean_pyro_ambient_temp, axis=1), c='blue', label='Pyrometer')
    ax[0].scatter(angles, np.mean(clean_tc_ambient_temp, axis=1),   c ='blue', marker='s', alpha=.7, label='Thermocouple')

    ax[0].set_title('Data')
    ax[0].set_xlabel('Viewing Angle (°)')
    ax[0].set_ylabel('Average Temperature (°C)')
    ax[0].set_ylim(16, 22)
    
    ax[1].plot(angles, np.mean((clean_pyro_ambient_temp - clean_tc_ambient_temp)**2, axis=1), c='blue', marker='.')
    
    ax[1].set_title('Error')
    ax[1].set_xlabel('Viewing Angle (°)')
    ax[1].set_ylabel('Mean Square Error')
    ax[1].set_ylim(0, 20)
    
    ax[0].legend()
    ax[0].grid(True)
    ax[1].grid(True)
    
    plt.show()
    
    fig, ax = plt.subplots(2, 1, layout='constrained', sharex=True)
    
    fig.suptitle('Elevated Temp. Clean Wall')
    
    ax[0].scatter(angles, np.mean(clean_pyro_hot_temp, axis=1), c='red', label='Pyrometer')
    ax[0].scatter(angles, np.mean(clean_tc_hot_temp, axis=1),   c ='red', marker='s', alpha=.7, label='Thermocouple')

    ax[0].set_title('Data')
    ax[0].set_xlabel('Viewing Angle (°)')
    ax[0].set_ylabel('Average Temperature (°C)')
    ax[0].set_ylim(100, 125)
    
    ax[1].plot(angles, np.mean((clean_pyro_hot_temp - clean_tc_hot_temp)**2, axis=1), c='red', marker='.')
    
    ax[1].set_title('Error')
    ax[1].set_xlabel('Viewing Angle (°)')
    ax[1].set_ylabel('Mean Square Error')
    ax[1].set_ylim(0, 250)
    
    ax[0].legend()
    ax[0].grid(True)
    ax[1].grid(True)
    plt.show()
    
    fig, ax = plt.subplots(2, 1, layout='constrained', sharex=True)
    
    fig.suptitle('Ambient Temp. Sooty Wall')
    
    ax[0].scatter(angles, np.mean(sooty_pyro_ambient_temp, axis=1), c='green', label='Pyrometer')
    ax[0].scatter(angles, np.mean(sooty_tc_ambient_temp, axis=1),   c ='green', marker='s', alpha=.7, label='Thermocouple')

    ax[0].set_title('Data')
    ax[0].set_xlabel('Viewing Angle (°)')
    ax[0].set_ylabel('Average Temperature (°C)')
    ax[0].set_ylim(16, 22)
    
    ax[1].plot(angles, np.mean((sooty_pyro_ambient_temp - sooty_tc_ambient_temp)**2, axis=1), c='green', marker='.')
    
    ax[1].set_title('Error')
    ax[1].set_xlabel('Viewing Angle (°)')
    ax[1].set_ylabel('Mean Square Error')
    ax[1].set_ylim(0, 20)
    
    ax[0].legend()
    ax[0].grid(True)
    ax[1].grid(True)
    
    plt.show()
    
    fig, ax = plt.subplots(2, 1, layout='constrained', sharex=True)
    
    fig.suptitle('Elevated Temp. Sooty Wall')
    
    ax[0].scatter(angles, np.mean(sooty_pyro_hot_temp, axis=1), c='purple', label='Pyrometer')
    ax[0].scatter(angles, np.mean(sooty_tc_hot_temp, axis=1),   c ='purple', marker='s', alpha=.7, label='Thermocouple')

    ax[0].set_title('Data')
    ax[0].set_xlabel('Viewing Angle (°)')
    ax[0].set_ylabel('Average Temperature (°C)')
    ax[0].set_ylim(100, 125)
    
    ax[1].plot(angles, np.mean((sooty_pyro_hot_temp - sooty_tc_hot_temp)**2, axis=1), c='purple', marker='.')
    
    ax[1].set_title('Error')
    ax[1].set_xlabel('Viewing Angle (°)')
    ax[1].set_ylabel('Mean Square error')
    ax[1].set_ylim(0, 250)
    
    ax[0].legend()
    ax[0].grid(True)
    ax[1].grid(True)
    plt.show()
    
    plt.show()


    return df

if __name__ == '__main__':
    
    if len(sys.argv) == 2:
        dir = sys.argv[1]
    else:
        dir = helper_functions.selectFolder()
        
    dir = Path(dir)
        
    flir_df = combineFlirData(dir / 'flir', force_update=False)
    pyro_df = combinePyroData(dir / 'pyro', force_update=True)
    

    
    


