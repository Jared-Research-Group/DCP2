import pandas as pd
import os
from functools import partial

from data_manipulation import selectFolder, dfToCsv
from batch_process import dataSearch

def splitPyrometerCsv(df, dir, ref='flir'):
    # if collection's data is already stored, continue
    if os.access(dir.path + '/pyrometer.csv', os.R_OK):
        return

    # assuming we are pulling time window from FLIR:
    if ref == 'flir':
        from thermography import getFrameData

        # get max, min timestamps from this specific data collection
        ts, _ = getFrameData(dir.path + '/FLIR', printFlag=False)

        startTime = min(ts)
        stopTime = max(ts)

        print(df)
        print(startTime)
        print(stopTime)

    # store pyrometer data bounded by startTime, stopTime in data collection folder
    windowed_df = df[(df['TimeStamp Adjust'] > startTime) & (df['TimeStamp Adjust'] < stopTime)].reset_index()
    dfToCsv(windowed_df, dir.path + '/pyrometer.csv')

    return

def getPyrometerData(dir):
    print('         Reading Pyrometer data...')

    df = pd.read_csv(dir + '/pyrometer.csv', parse_dates=['timestamp', 'TimeStamp Adjust'])

    return df['TimeStamp Adjust'], df['rolling_avg_C']



if __name__ == '__main__':
    parent_dir = selectFolder()

    df = pd.read_csv(parent_dir + '/pyrometer.csv', parse_dates=['timestamp', 'TimeStamp Adjust'])
    dataSearch(parent_dir, partial(splitPyrometerCsv, df))