import pandas as pd
import sys
from tkinter import filedialog
from lembox_visualization import getLemboxData
from position import readRSI, plotPosValColormap, plotPos
from audio_time_scale import mic_time
from data_manipulation import dfToCsv, getStartStop, dfHasColumn, getRollingAvg, quickPlot
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import math

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

    if noAlignment or incompAlignment or forceDataUpdate:                       # if (for any reason) aligned_data.csv is incomplete, rebuild it

        # collect required data from each datatype-specific program

        #interpolated lembox timestamps, raw current, raw voltage, rolling average current, rolling average voltage, global timestamp of data collection start
        lem_time, curr, volt, avgI, avgV = getLemboxData(dir + '/lembox_data.csv', 5000, forceDataUpdate)
        lem = [lem_time, curr, avgI, volt, avgV]

        # position data, calculated velocity data, interpolated rsi timestamps, global timestamp of data collection start, calculated RSI sample rate
        pos, vel, rsi_time, rsiCalcSR = readRSI(dir + '/robot_data.csv', 1000, forceDataUpdate)
        rsi = [rsi_time, pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], vel[3]]
        sample_rates['rsi'] = rsiCalcSR

        # interpolated mic timestamps, amplitude, global timestamp of data collection start
        mic_t, mic_A = mic_time(dir + '/microphone_data.csv', dir + '/microphone_data_aligned.csv', sample_rate = 48000)
        mic = [mic_t, mic_A]

        nonaligned_data = {'lembox':lem, 'rsi':rsi, 'mic':mic}

        alignmentDataset = 'mic'
        if nonaligned_data['lembox'][0][-1] - nonaligned_data['mic'][0][-1] > 2: alignmentDataset = 'lembox'                            # we somehow lost microphone data during the weld
        aligned_data = {'lembox':[], 'rsi':[], 'mic':[]}

        for k in aligned_data.keys():
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
    
        df = pd.DataFrame({'time':nonaligned_data[alignmentDataset][0], 'Amplitude':aligned_data['mic'][0], 'Current(A)': aligned_data['lembox'][0], 'Avg_Current(A)': aligned_data['lembox'][1], 
                            'Voltage(V)': aligned_data['lembox'][2], 'Avg_Voltage(V)': aligned_data['lembox'][3], 'Pos_x(mm)': aligned_data['rsi'][0], 
                            'Pos_y(mm)': aligned_data['rsi'][1], 'Pos_z(mm)': aligned_data['rsi'][2], 'Vel_x(mm/s)': aligned_data['rsi'][3],
                            'Vel_y(mm/s)': aligned_data['rsi'][4], 'Vel_z(mm/s)': aligned_data['rsi'][5], 'Vel_Comb(mm/s)': aligned_data['rsi'][6]})
       
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