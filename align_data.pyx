import pandas as pd
import sys
from tkinter import filedialog
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import math

from lembox import getLemboxData
from core_scripts.position import readRSI, plotPosValColormap, plotPos
from audio_time_scale import mic_time
from thermography import getFrameData
from thermocouple import getThermocoupleData

from data_manipulation import dfToCsv, getStartStop, dfHasColumn, getRollingAvg, quickPlot

import cython

sample_rates = {'lembox':20000, 'mic':48000, 'rsi':250}                         # Hz                        (rsi rate may present as 1000 Hz, known bug)
expected_columns = ['Current(A)', 'Pos_x(mm)', 'Amplitude']                     # 1 per raw data *.csv


def alignData(dir, forceDataUpdate=False):
    print(f'    Aligning data for {dir}...')

    noAlignment = not os.access(dir +'/aligned_data.csv', os.R_OK)              # does aligned_data.csv exist?

    if not noAlignment:                                                         # if aligned_data.csv exists, does it contain data from all expected sources?
        df = pd.read_csv(dir +'/aligned_data.csv')

        incompAlignment = False
        for c in expected_columns:
            if not dfHasColumn(df, c): incompAlignment = True

    cdef list lem, rsi, mic, flir
    cdef dict nonaligned_data, aligned_data

    nonaligned_data = {}
    aligned_data = {}
    if noAlignment or incompAlignment or forceDataUpdate:                       # if (for any reason) aligned_data.csv is incomplete, rebuild it

        # collect required data from each datatype-specific program

        try:
            tc_dat = getThermocoupleData(dir)
            tc_t = tc_dat['Relative Time (s)']
            
            nonaligned_data['tc'] = [tc_dat['Relative Time (s)'], tc_dat['Channel 0 (°C)'], tc_dat['Channel 1 (°C)'], tc_dat['Channel 2 (°C)'], tc_dat['Channel 3 (°C)']]
            aligned_data['tc'] = []

            alignmentDataset = 'tc'
        except FileNotFoundError: [print('      thermocouple_data.csv not found.')]

        try:
            flir_t, flir_path = getFrameData(dir + '/FLIR')

            flir_init = flir_t[0]
            for i, t in enumerate(flir_t):
                flir_t[i] = t - flir_init

            nonaligned_data['flir'] = [flir_t, flir_path]
            aligned_data['flir'] = []

            alignmentDataset = 'flir'
        except FileNotFoundError: print('       FLIR data not found.')

        # position data, calculated velocity data, interpolated rsi timestamps, global timestamp of data collection start, calculated RSI sample rate
        try:
            pos, vel, rsi_time, rsiCalcSR = readRSI(dir + '/robot_data.csv', 1000, forceDataUpdate)
            sample_rates['rsi'] = rsiCalcSR

            nonaligned_data['rsi'] = [rsi_time, pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], vel[3]]
            aligned_data['rsi'] = []

            alignmentDataset = 'rsi'
        except FileNotFoundError: print('       robot_data.csv not found.')        

        #interpolated lembox timestamps, raw current, raw voltage, rolling average current, rolling average voltage, global timestamp of data collection start
        try:
            lem_time, curr, volt, avgI, avgV = getLemboxData(dir + '/lembox_data.csv', 5000, forceDataUpdate)

            nonaligned_data['lembox'] = [lem_time, curr, avgI, volt, avgV]
            aligned_data['lembox'] = []

            alignmentDataset = 'lembox'
        except FileNotFoundError: print('       lembox_data.csv not found.')

        # interpolated mic timestamps, amplitude, global timestamp of data collection start
        try:
            mic_t, mic_A = mic_time(dir + '/microphone_data.csv', dir + '/microphone_data_aligned.csv', sample_rate = 48000)
            mic = [mic_t, mic_A]

            nonaligned_data['mic'] = mic
            aligned_data['mic'] = []

            alignmentDataset = 'mic'
        except FileNotFoundError: print('       microphone_data.csv not found.')

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
                    'flir':['FLIR_frame']}

        df = pd.DataFrame({'time':nonaligned_data[alignmentDataset][0]})
        for dat in aligned_data.keys():
            for i, stream in enumerate(aligned_data[dat]):
                df[labels[dat][i]] = stream
       
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