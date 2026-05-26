import matplotlib.pyplot as plt
import pandas as pd
import time
import math
import os
import sys
from data_manipulation import selectFolder
from batch_process     import dataSearch

# read raw thermocouple data *.csv as pandas DataFrame
def getThermocoupleData(d, filename='thermocouple_data.csv'):

    print('         Reading thermocouple data...')
    df = pd.read_csv(d + '/' + filename, encoding='cp1252', parse_dates=['Timestamp'])         # weird specific encoding required for successful read

    return df

# plot all 4 thermocouple channels as overlayed series in single axis
def plotThermocouple(df):
    t = df['Timestamp']

    # build floats representing epoch time in seconds from timestamp text in df
    timestamps = []
    for i, timestamp in enumerate(t):
        timestamp_micro = float(timestamp[-6:])/math.pow(10,6)
        timestamp = time.mktime(time.strptime(timestamp[:-7], '%Y-%m-%d %H:%M:%S'))
        timestamp += timestamp_micro

        timestamps.append(timestamp)

    # convert from epoch time to time since recording started
    startTime = timestamps[0]
    for i, t in enumerate(timestamps):
        timestamps[i] = t - startTime

    # pick out data from 
    temp = []
    for i in range(4):
        channel = 'Channel ' + str(i) + ' (°C)'
        temp.append(df[channel])

    fig, ax = plt.subplots(layout='constrained')

    colors = ['#0d6cbf', '#9d16db','#db2a16', '#16db19']

    for i in range(4):
        ax.scatter(timestamps, temp[i], s=2, c=colors[i], label='Channel ' + str(i))

    ax.set_ylabel('Temperature (°C)')
    ax.set_xlabel('Time (s)')
    ax.legend(markerscale=3)

    plt.show()


    return

def main():

    if len(sys.argv) == 3:
        dir = sys.argv[1]
        fname = sys.argv[2]

        d = getThermocoupleData(dir, fname)
        plotThermocouple(d)
    else:
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