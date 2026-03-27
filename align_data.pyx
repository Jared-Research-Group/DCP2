import pandas as pd
import sys
from tkinter import filedialog
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import math
from datetime import datetime, timedelta

from lembox import getLemboxData
from core_scripts.position import readRSI, plotPosValColormap, plotPos
from audio_time_scale import mic_time
from thermography import getFrameData as getFrameData_FLIR
from thermocouple import getThermocoupleData
from xiris import getFrameList as getFrameData_XIR

from data_manipulation import dfToCsv, getStartStop, dfHasColumn, getRollingAvg, quickPlot

import cython

sample_rates = {'lembox':20000, 'mic':48000, 'rsi':250}                         # Hz                        (rsi rate may present as 1000 Hz, known bug)
expected_columns = ['Current(A)', 'Pos_x(mm)', 'Amplitude']                     # 1 per raw data *.csv


def alignData(dir, forceDataUpdate=False):
    print(f'    Aligning data for {dir}...')

    noAlignment = not os.access(dir +'/aligned_data.csv', os.R_OK)              # does aligned_data.csv exist?

    if not noAlignment:                                                         # if aligned_data.csv exists, does it contain data from all expected sources?
        df = pd.read_csv(dir +'/aligned_data.csv', parse_dates=['time'])

        incompAlignment = False
        #for c in expected_columns:                                             # program is now general for any combination of data streams. This check would force unnessecary overwrites
        #    if not dfHasColumn(df, c): incompAlignment = True

    cdef dict nonaligned_data, aligned_data

    nonaligned_data = {}
    aligned_data = {}
    if noAlignment or incompAlignment or forceDataUpdate:                       # if (for any reason) aligned_data.csv is incomplete, rebuild it

        # collect required data from each datatype-specific program. data types are collected in ascending order of data rate. This way, the highest data rate
        # (last non-exception try block) is always set as alignmentDataset

        ############################################
        # Thermocouple

        try:
            tc_dat = getThermocoupleData(dir)

            tc_t = []
            for t in tc_dat['Timestamp']:
                tc_t.append(datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f'))
            
            nonaligned_data['tc'] = [tc_t, tc_dat['Channel 0 (°C)'], tc_dat['Channel 1 (°C)'], tc_dat['Channel 2 (°C)'], tc_dat['Channel 3 (°C)']]
            aligned_data['tc'] = []

            alignmentDataset = 'tc'
        except FileNotFoundError: [print('      thermocouple_data.csv not found.')]

        ############################################
        # FLIR

        try:
            flir_t, flir_path = getFrameData_FLIR(dir + '/FLIR')

            nonaligned_data['flir'] = [flir_t, flir_path]
            aligned_data['flir'] = []

            alignmentDataset = 'flir'
        except FileNotFoundError: print('       FLIR data not found.')

        ############################################
        # XIR

        try:
            xir_path = getFrameData_XIR(dir + '/Xiris')

            xir_t = []
            for path in xir_path:
                t = os.path.split(path)[1][:-4]
                
                time = datetime.fromtimestamp(int(t[:-6]))
                time += timedelta(microseconds=int(t[-6:]))

                xir_t.append(time)

            nonaligned_data['xir'] = [xir_t, xir_path]
            aligned_data['xir'] = []

            alignmentDataset = 'xir'
        except FileNotFoundError: print('       Xiris data not found.')

        ############################################

        # RSI

        # position data, calculated velocity data, interpolated rsi timestamps, global timestamp of data collection start, calculated RSI sample rate
        try:
            pos, vel, rsi_timestamps, rsi_time, rsiCalcSR = readRSI(dir + '/robot_data.csv', 1000, forceDataUpdate)
            sample_rates['rsi'] = rsiCalcSR

            nonaligned_data['rsi'] = [rsi_timestamps, pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], vel[3]]
            aligned_data['rsi'] = []

            alignmentDataset = 'rsi'
        except FileNotFoundError: print('       robot_data.csv not found.')       

        ############################################
        # LEMBOX

        #interpolated lembox timestamps, raw current, raw voltage, rolling average current, rolling average voltage, global timestamp of data collection start
        try:

            lem_time_rel, curr, volt, avgI, avgV, lem_time = getLemboxData(dir + '/lembox_data.csv', 5000, forceDataUpdate)

            nonaligned_data['lembox'] = [lem_time, curr, avgI, volt, avgV]
            aligned_data['lembox'] = []

            alignmentDataset = 'lembox'
        except FileNotFoundError: print('       lembox_data.csv not found.')

        ############################################
        # MIC

        # interpolated mic timestamps, amplitude, global timestamp of data collection start
        try:
            mic_t, mic_A = mic_time(dir + '/microphone_data.csv', dir + '/microphone_data_aligned.csv', sample_rate = 48000)
            mic = [mic_t, mic_A]

            nonaligned_data['mic'] = mic
            aligned_data['mic'] = []

            alignmentDataset = 'mic'
        except FileNotFoundError: print('       microphone_data.csv not found.')

        ############################################

        # find the latest start time of our data streams. We will drop all data before this time later
        lastStart = datetime(1990, 1, 1)
        for key, dat in nonaligned_data.items():
            if dat[0][0] > lastStart:
                lastStart = dat[0][0]

        # Do the actual alignment

        # for each data source in need of alignment
        for k in aligned_data.keys():

            # for each series in dataset
            for i in range(len(nonaligned_data[k]) - 1): 
                aligned_data[k].append([])                     # size aligned_data to size(number of data columns in non-basis timescale)
            
            if k == alignmentDataset:
                for i in range(len(nonaligned_data[k]) - 1):
                    aligned_data[k][i] = nonaligned_data[k][i+1]
                continue
            
            index = 0
            for t in nonaligned_data[alignmentDataset][0]:
                while t > nonaligned_data[k][0][index] and index < len(nonaligned_data[k][0]) - 1: index += 1

                for i in range(len(nonaligned_data[k]) - 1):
                    aligned_data[k][i].append(nonaligned_data[k][i+1][index])

        labels = {'tc':['Channel_0(°C)', 'Channel_1(°C)', 'Channel_2(°C)', 'Channel_3(°C)'], 'mic':['Amplitude'], \
                    'lembox':['Current(A)', 'Avg_Current(A)', 'Voltage(V)', 'Avg_Voltage(V)'], 'rsi':['Pos_x(mm)', \
                    'Pos_y(mm)', 'Pos_z(mm)', 'Vel_x(mm/s)', 'Vel_y(mm/s)', 'Vel_z(mm/s)', 'Vel_comb(mm/s)'], \
                    'flir':['FLIR_frame'], 'xir':['Xiris_frame']}

        df = pd.DataFrame({'time':nonaligned_data[alignmentDataset][0]})
        for dat in aligned_data.keys():
            for i, stream in enumerate(aligned_data[dat]):
                df[labels[dat][i]] = stream

        df = df[df['time'] >= lastStart]        # drop all data before latest datastream start. We do it after alignment because it is easier to drop from a dataframe than individual data streams. Optimizaion to be had here.

        t_rel = []
        startTime = df.loc[0, 'time']
        for t in df['time']:
            t_rel.append(t - startTime)

        df['Relative_Time(s)'] = pd.Series(t_rel)

        dfToCsv(df, dir + '/aligned_data.csv')

    return df

def main():
    if len(sys.argv) != 2:
        dir = filedialog.askdirectory()
    else:
        dir = sys.argv[1]
    
    df = alignData(dir, True)
    
    startTime, stopTime = getStartStop(df['Avg_Voltage(V)'], 1)
    #startTime += 2 * sample_rates['mic'] ; stopTime -= 2 * sample_rates['mic']

    #startTime += sample_rates['mic']*2
    #stopTime = startTime + int(sample_rates['mic']*.25)

    #data = [[[df['Pos_y(mm)'][startTime:stopTime], df['Voltage(V)'][startTime:stopTime]], [df['Pos_y(mm)'][startTime:stopTime], df['Current(A)'][startTime:stopTime]], [df['Pos_y(mm)'][startTime:stopTime], abs(df['Amplitude'][startTime:stopTime])]]]
    #quickPlot(data)

    plotPosValColormap((df['Pos_x(mm)'][startTime:stopTime], df['Pos_y(mm)'][startTime:stopTime], df['Pos_z(mm)'][startTime:stopTime]), df['Avg_Current(A)'][startTime:stopTime], 'Rolling Average Current (A)', 'Current as a function of position')
    plt.savefig(dir + '/visualizations/current_3d.png')
    plotPosValColormap((df['Pos_x(mm)'][startTime:stopTime], df['Pos_y(mm)'][startTime:stopTime], df['Pos_z(mm)'][startTime:stopTime]), df['Avg_Voltage(V)'][startTime:stopTime], 'Rolling Average Voltage (V)', 'Voltage as a function of position')
    plt.savefig(dir + '/visualizations/voltage_3d.png')
    plotPosValColormap((df['Pos_x(mm)'][startTime + int(0.05 * sample_rates['mic']):stopTime], df['Pos_y(mm)'][startTime + int(0.05 * sample_rates['mic']):stopTime], df['Pos_z(mm)'][startTime + int(0.05 * sample_rates['mic']):stopTime]), pd.Series(getRollingAvg((df['Amplitude'][startTime:stopTime]), int(0.05 * sample_rates['mic']))), 'Amplitude', 'Amplitude as a function of position', 0, 0.0001)
    plt.savefig(dir + '/visualizations/amplitude_3d.png')

    #print(np.nanmax(getRollingAvg(df['Amplitude'][startTime:stopTime], int(0.05 * sample_rates['mic']))))
    
if __name__ == '__main__': main()