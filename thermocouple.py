import matplotlib.pyplot as plt
import pandas as pd
import time
import math
import os
from data_manipulation import selectFolder
from batch_process     import dataSearch

def getThermocoupleData(d):

    df = pd.read_csv(d + '/thermocouple_data.csv', encoding='cp1252')

    return df

def plotThermocouple(df):
    t = df['Timestamp']

    timestamps = []
    for i, timestamp in enumerate(t):
        timestamp_micro = float(timestamp[-6:])/math.pow(10,6)
        timestamp = time.mktime(time.strptime(timestamp[:-7], '%Y-%m-%d %H:%M:%S'))
        timestamp += timestamp_micro

        timestamps.append(timestamp)


    temp = []
    for i in range(4):
        channel = 'Channel ' + str(i) + ' (°C)'
        temp.append(df[channel])

    fig, ax = plt.subplots(2,2, layout='constrained', sharex=True)

    for i in range(2):
        for j in range(2):
            ax[i][j].scatter(timestamps, temp[i*2 + j], s=2)
            ax[i][j].set_ylabel('Temperature (°C)')
            ax[i][j].set_title('Channel ' + str(i*2 + j))
        ax[1][i].set_xlabel('Time (s)')

    plt.show()


    return

def main():
    dir = selectFolder()

    if os.path.split(dir)[1].startswith('data_collection'):
        d = getThermocoupleData(dir)
        plotThermocouple(d)
        return
    
    dat = []
    def dataCallback(e):
        d = getThermocoupleData(e.path)
        dat.append(d)
        return

    dataSearch(dir, dataCallback, printFlag=False)
    df = dat.pop()
    while dat:
        df = pd.concat([dat.pop(), df], ignore_index=True)

    plotThermocouple(df)

    return

if __name__ == '__main__': main()