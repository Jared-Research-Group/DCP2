import pandas as pd
import sys
from tkinter import filedialog
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import math
from datetime import datetime, timedelta, timezone

from lembox import getLemboxData
from core_scripts.position import readRSI, plotPosValColormap, plotPos
from audio_time_scale import mic_time
from thermography import getFrameData as getFrameData_FLIR
from thermocouple import getThermocoupleData
from xiris import getFrameList as getFrameData_XIR
from pyrometer import getPyrometerData

from data_manipulation import dfToCsv, getStartStop, dfHasColumn, getRollingAvg, quickPlot

import cython

sample_rate = {'lembox':20 * 10**3, 'mic':48 * 10**3, 'rsi':250, 'flir':30, 'tc':3.5, 'py':2}                         # Hz                        (rsi rate may present as 1000 Hz, known bug)
types = {'time':float, 'Pyrometer_Temp(°C)':float, 'Channel_0(°C)':float, 'Channel_1(°C)':float, 'Channel_2(°C)':float, 'Channel_3(°C)':float, 'Amplitude':float, \
                    'Current(A)':float, 'Avg_Current(A)':float, 'Voltage(V)':float, 'Avg_Voltage(V)':float, 'Pos_x(mm)':float, \
                    'Pos_y(mm)':float, 'Pos_z(mm)':float, 'Vel_x(mm/s)':float, 'Vel_y(mm/s)':float, 'Vel_z(mm/s)':float, 'Vel_comb(mm/s)':float, \
                    'FLIR_frame':str, 'Xiris_frame':str}


def getCutoffIndex(id, start_time, last_sensor_time):
    sr = sample_rate[id]

    lost_time = (last_sensor_time - start_time).total_seconds()
    
    return int(math.ceil(lost_time*sr))

def interpTime(id, len):
    sr = sample_rate[id]
    p  = 1/sr

    return np.arange(0, p*len, p)[:len] 


def alignData(dir, forceDataUpdate=False):
    print(f'    Aligning data for {dir}...')

    noAlignment = not os.access(dir +'/aligned_data.csv', os.R_OK)              # does aligned_data.csv exist?

    if not noAlignment:                                                         # if aligned_data.csv exists, does it contain data from all expected sources?
        df = pd.read_csv(dir +'/aligned_data.csv', dtype=types)

    cdef dict nonaligned_data, aligned_data

    nonaligned_data = {}
    aligned_data = {}
    if noAlignment or forceDataUpdate:                       # if (for any reason) aligned_data.csv is incomplete, rebuild it

        # collect required data from each datatype-specific program. data types are collected in ascending order of data rate. This way, the highest data rate
        # (last non-exception try block) is always set as alignmentDataset

        start_times = {}
        abs_times = {}

        ############################################
        # Pyrometer

        try:
            py_time, py_temp = getPyrometerData(dir)

            nonaligned_data['py'] = [[], py_temp.to_numpy()]
            aligned_data['py'] = []

            start_times['py'] = py_time[0]
            abs_times['py'] = py_time - py_time[0]

            alignmentDataset = 'py'
        except FileNotFoundError: [print('      pyrometer.csv not found.')]


        ############################################
        # Thermocouple

        try:
            tc_dat = getThermocoupleData(dir)

            tc_t = []
            
            nonaligned_data['tc'] = [tc_t, tc_dat.iloc[:, 2].to_numpy(), tc_dat.iloc[:, 3].to_numpy(), tc_dat.iloc[:, 4].to_numpy(), tc_dat.iloc[:, 5].to_numpy()]
            aligned_data['tc'] = []

            start_times['tc'] = tc_dat.loc[0, 'Timestamp']
            abs_times['tc'] = (tc_dat['Timestamp'] - tc_dat.loc[0, 'Timestamp']).to_list()

            alignmentDataset = 'tc'
        except FileNotFoundError: [print('      thermocouple_data.csv not found.')]

        ############################################
        # FLIR

        try:
            flir_t, flir_path = getFrameData_FLIR(dir + '/FLIR')

            nonaligned_data['flir'] = [[], flir_path]
            aligned_data['flir'] = []

            start_times['flir'] = flir_t[0]
            abs_times['flir'] = flir_t

            alignmentDataset = 'flir'
        except FileNotFoundError: print('       FLIR data not found.')

        ############################################
        # XIR

            # Xiris doesn't have a consistent sample rate, so we can't interpolate it's timestamps

        try:
            xir_path = getFrameData_XIR(dir + '/Xiris')

            xir_t = []
            xir_a_time = []

            xir_t_start = 0
            for path in xir_path:
                t = os.path.split(path)[1][:-4]
                
                time = datetime.fromtimestamp(int(t[:-6]))
                time += timedelta(microseconds=int(t[-6:]))

                xir_a_time.append(time)

                if xir_t_start == 0: xir_t_start = time

                time = (time - xir_t_start).total_seconds()

                xir_t.append(time)

            nonaligned_data['xir'] = [xir_t, xir_path]
            aligned_data['xir'] = []

            start_times['xir'] = xir_a_time[0]
            abs_times['xir'] = xir_a_time

            alignmentDataset = 'xir'
        except FileNotFoundError: print('       Xiris data not found.')

        ############################################
        # RSI

        # position data, calculated velocity data, interpolated rsi timestamps, global timestamp of data collection start, calculated RSI sample rate
        try:
            pos, vel, rsi_timestamps, rsi_time, rsiCalcSR = readRSI(dir + '/robot_data.csv', 1000, forceDataUpdate)
            sample_rate['rsi'] = rsiCalcSR

            nonaligned_data['rsi'] = [[], pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], vel[3]]
            aligned_data['rsi'] = []

            start_times['rsi'] = rsi_timestamps[0]
            abs_times['rsi'] = rsi_timestamps

            alignmentDataset = 'rsi'
        except FileNotFoundError: print('       robot_data.csv not found.')       

        ############################################
        # LEMBOX

        #interpolated lembox timestamps, raw current, raw voltage, rolling average current, rolling average voltage, global timestamp of data collection start
        try:

            lem_time_rel, curr, volt, avgI, avgV, lem_time = getLemboxData(dir + '/lembox_data.csv', 5000, forceDataUpdate)

            nonaligned_data['lembox'] = [[], curr, avgI, volt, avgV]
            aligned_data['lembox'] = []

            start_times['lembox'] = lem_time[0]
            abs_times['lembox'] = lem_time

            alignmentDataset = 'lembox'
        except FileNotFoundError: print('       lembox_data.csv not found.')

        ############################################
        # MIC

        # interpolated mic timestamps, amplitude, global timestamp of data collection start
        try:
            mic_t, mic_A = mic_time(dir + '/microphone_data.csv', dir + '/microphone_data_aligned.csv', sample_rate = 48000)
            mic = [[], mic_A]

            nonaligned_data['mic'] = mic
            aligned_data['mic'] = []

            start_times['mic'] = mic_t[0]
            abs_times['mic'] = mic_t

            alignmentDataset = 'mic'
        except FileNotFoundError: print('       microphone_data.csv not found.')

        ############################################

        # we need to detect and resolve an issue in which LEMBOX timestamps are (only sometimes) recorded in UTC, whereas all other data types record always in EST

        # THIS NEEDS TO MOVE TO LEMBOX.PYX
        if 'lembox' in nonaligned_data.keys():
            i = 0
            while list(start_times.keys())[i] == 'lembox': i += 1
            lembox_hour_offset = timedelta(hours=round((start_times['lembox'] - list(start_times.items())[i][1]).total_seconds() / 3600))

            start_times['lembox'] -= lembox_hour_offset

        # find the latest start time of our data streams. We will drop all data before this time later
        lastStart = datetime(1990, 1, 1)
        for key, dat in start_times.items():
            #print(key + ': ' + str(type(dat[0][0])))
            if dat > lastStart:
                lastStart = dat

        print(lastStart)

        # generate interpolated timestamps for all data but XIRIS
        for key, dat in nonaligned_data.items():
            if not dat[0]:
                start_index = getCutoffIndex(key, start_times[key], lastStart)
                
                # Slice all data arrays to remove data before lastStart
                for i in range(len(nonaligned_data[key])):
                    nonaligned_data[key][i] = nonaligned_data[key][i][start_index:]
                
                # Generate relative timestamps starting from 0
                nonaligned_data[key][0] = interpTime(key, len(nonaligned_data[key][1]))

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
            
            index = 1
            one_past_index = len(nonaligned_data[k][0])
            non_al_time = nonaligned_data[k][0]
            non_al_time = np.append(non_al_time, np.inf)

            print('key: ', k)
            for t in nonaligned_data[alignmentDataset][0]:

                while index < one_past_index and t >= non_al_time[index]: 
                    index += 1

                for i in range(len(nonaligned_data[k]) - 1):
                        aligned_data[k][i].append(nonaligned_data[k][i+1][index - 1])

        labels = {'py':['Pyrometer_Temp(°C)'], 'tc':['Channel_0(°C)', 'Channel_1(°C)', 'Channel_2(°C)', 'Channel_3(°C)'], 'mic':['Amplitude'], \
                    'lembox':['Current(A)', 'Avg_Current(A)', 'Voltage(V)', 'Avg_Voltage(V)'], 'rsi':['Pos_x(mm)', \
                    'Pos_y(mm)', 'Pos_z(mm)', 'Vel_x(mm/s)', 'Vel_y(mm/s)', 'Vel_z(mm/s)', 'Vel_comb(mm/s)'], \
                    'flir':['FLIR_frame'], 'xir':['Xiris_frame']}

        df = pd.DataFrame({'time':nonaligned_data[alignmentDataset][0]})
        for dat in aligned_data.keys():
            for i, stream in enumerate(aligned_data[dat]):
                    df[labels[dat][i]] = stream[-len(df['time']):]

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

    #plotPosValColormap((df['Pos_x(mm)'][startTime:stopTime], df['Pos_y(mm)'][startTime:stopTime], df['Pos_z(mm)'][startTime:stopTime]), df['Avg_Current(A)'][startTime:stopTime], 'Rolling Average Current (A)', 'Current as a function of position')
    #plt.savefig(dir + '/visualizations/current_3d.png')
    #plotPosValColormap((df['Pos_x(mm)'][startTime:stopTime], df['Pos_y(mm)'][startTime:stopTime], df['Pos_z(mm)'][startTime:stopTime]), df['Avg_Voltage(V)'][startTime:stopTime], 'Rolling Average Voltage (V)', 'Voltage as a function of position')
    #plt.savefig(dir + '/visualizations/voltage_3d.png')
    #plotPosValColormap((df['Pos_x(mm)'][startTime + int(0.05 * sample_rates['mic']):stopTime], df['Pos_y(mm)'][startTime + int(0.05 * sample_rates['mic']):stopTime], df['Pos_z(mm)'][startTime + int(0.05 * sample_rates['mic']):stopTime]), pd.Series(getRollingAvg((df['Amplitude'][startTime:stopTime]), int(0.05 * sample_rates['mic']))), 'Amplitude', 'Amplitude as a function of position', 0, 0.0001)
    #plt.savefig(dir + '/visualizations/amplitude_3d.png')

    #print(np.nanmax(getRollingAvg(df['Amplitude'][startTime:stopTime], int(0.05 * sample_rates['mic']))))
    
if __name__ == '__main__': main()