import pandas as pd
import sys
from tkinter import filedialog
from lembox_visualization import getLemboxData
from position import readRSI, plotPosValColormap
from audio_time_scale import mic_time
from data_manipulation import dfToCsv, getStartStop, dfHasColumn, getRollingAvg, quickPlot
import os
import matplotlib.pyplot as plt
import numpy as np

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

        lem_time, curr, volt, avgI, avgV = getLemboxData(dir + '/lembox_data.csv', 1000, forceDataUpdate)
        lem = (lem_time, curr, avgI, volt, avgV)

        pos, vel, rsi_time = readRSI(dir + '/robot_data.csv', 1, forceDataUpdate)
        rsi = (pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], vel[3], rsi_time)
    
        mic_t, mic_A = mic_time(dir + '/microphone_data.csv', dir + '/microphone_data_aligned.csv', sample_rate = 48000)
        mic = (mic_t, mic_A)

        aligned_data = ()
        for i in range(len(rsi) - 1): aligned_data += ([],)                     # size aligned_data to size(number of data columns in non-basis timescale)

        rsi_index = 0
        for t in mic[0]:
            while t > rsi[-1][rsi_index] and rsi_index < len(rsi[-1]) - 1: rsi_index += 1

            for i in range(len(rsi) - 1):
                aligned_data[i].append(rsi[i][rsi_index])
        
        aligned_data_lem = ()
        for j in range(len(lem)): aligned_data_lem += ([],)

        lem_index = 0 
        for l in mic[0]:
            while l > lem[0][lem_index] and lem_index < len(lem[0]) - 1: lem_index += 1

            for j in range(len(lem)):
                aligned_data_lem[j].append(lem[j][lem_index])

    
        df = pd.DataFrame({'time':mic[0], 'Amplitude':mic[1], 'Current(A)': aligned_data_lem[1], 'Avg_Current(A)': aligned_data_lem[2], 
                            'Voltage(V)': aligned_data_lem[3], 'Avg_Voltage(V)': aligned_data_lem[4], 'Pos_x(mm)': aligned_data[0], 
                            'Pos_y(mm)': aligned_data[1], 'Pos_z(mm)': aligned_data[2], 'Vel_x(mm/s)': aligned_data[3],
                            'Vel_y(mm/s)': aligned_data[4], 'Vel_z(mm/s)': aligned_data[5], 'Vel_Comb(mm/s)': aligned_data[6]})
       
        dfToCsv(df, dir + '/aligned_data.csv')

    return df

def main():
    if len(sys.argv) != 2:
        dir = filedialog.askdirectory()
    else:
        dir = sys.argv[1]
    
    df = alignData(dir, False)
    
    startTime, stopTime = getStartStop(df['Avg_Voltage(V)'], 1)
    #startTime += 2 * sample_rates['mic'] ; stopTime -= 2 * sample_rates['mic']

    #startTime += sample_rates['mic']*2
    #stopTime = startTime + int(sample_rates['mic']*.25)

    #data = [[[df['Pos_y(mm)'][startTime:stopTime], df['Voltage(V)'][startTime:stopTime]], [df['Pos_y(mm)'][startTime:stopTime], df['Current(A)'][startTime:stopTime]], [df['Pos_y(mm)'][startTime:stopTime], abs(df['Amplitude'][startTime:stopTime])]]]
    #quickPlot(data)


    #plt.show()

    plotPosValColormap((df['Pos_x(mm)'][startTime:stopTime], df['Pos_y(mm)'][startTime:stopTime], df['Pos_z(mm)'][startTime:stopTime]), df['Avg_Current(A)'][startTime:stopTime], 'Rolling Average Current (A)', 'Current as a function of position')
    plt.savefig(dir + '/visualizations/current_3d.png')
    plotPosValColormap((df['Pos_x(mm)'][startTime:stopTime], df['Pos_y(mm)'][startTime:stopTime], df['Pos_z(mm)'][startTime:stopTime]), df['Avg_Voltage(V)'][startTime:stopTime], 'Rolling Average Voltage (V)', 'Voltage as a function of position')
    plt.savefig(dir + '/visualizations/voltage_3d.png')
    plotPosValColormap((df['Pos_x(mm)'][startTime + int(0.05 * sample_rates['mic']):stopTime], df['Pos_y(mm)'][startTime + int(0.05 * sample_rates['mic']):stopTime], df['Pos_z(mm)'][startTime + int(0.05 * sample_rates['mic']):stopTime]), pd.Series(getRollingAvg((df['Amplitude'][startTime:stopTime]), int(0.05 * sample_rates['mic']))), 'Amplitude', 'Amplitude as a function of position', 0, 0.003)
    plt.savefig(dir + '/visualizations/amplitude_3d.png')

    #print(np.nanmax(getRollingAvg(df['Amplitude'][startTime:stopTime], int(0.05 * sample_rates['mic']))))
    
if __name__ == '__main__': main()