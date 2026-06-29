import pandas as pd
import os
from functools import partial
from pathlib import Path
from datetime import timedelta

from data_manipulation import selectFolder, dfToCsv
from batch_process import dataSearch

def splitPyrometerCsv(df, dir, ref='thermocouple'):
    dir = Path(dir)
    
    # if collection's data is already stored, continue
    if os.access(dir / 'pyrometer.csv', os.R_OK):
        return

    # assuming we are pulling time window from FLIR:
    if ref == 'flir':
        from thermography import getFrameData

        # get max, min timestamps from this specific data collection
        ts, _ = getFrameData(dir / 'FLIR', printFlag=False)

        startTime = min(ts)
        stopTime = max(ts)

        print(df)
        print(startTime)
        print(stopTime)
        
    elif ref == 'thermocouple':
        from thermocouple import getThermocoupleData
        
        tc = getThermocoupleData(dir)
        
        startTime = tc['Timestamp'].min()
        stopTime  = tc['Timestamp'].max()

    # store pyrometer data bounded by startTime, stopTime in data collection folder
    if 'TimeStamp Adjust' in df.columns:
        windowed_df = df[(df['TimeStamp Adjust'] > startTime) & (df['TimeStamp Adjust'] < stopTime)].reset_index()
    
    else:
        windowed_df = df[(df['timestamp'] + timedelta(hours=1) > startTime) & (df['timestamp'] + timedelta(hours=1) < stopTime)].reset_index()
    
    windowed_df.to_csv(dir / 'pyrometer.csv', index=False)

    return

def getPyrometerData(dir):
    print('         Reading Pyrometer data...')
    
    date_cols = ['timestamp']
    check_cols = pd.read_csv(dir / 'pyrometer.csv')
    if 'TimeStamp Adjust' in check_cols.columns:
        date_cols.append('TimeStamp Adjust')

    df = pd.read_csv(dir / 'pyrometer.csv', parse_dates=date_cols)
    df.reset_index(inplace=True)

    if 'TimeStamp Adjust' in df.columns:
        return df['TimeStamp Adjust'], df['rolling_avg_C']
    else:
        return df['timestamp'] + timedelta(hours=1), df['rolling_avg_C']



if __name__ == '__main__':
    parent_dir = selectFolder()

    date_cols = ['timestamp']
    check_cols = pd.read_csv(parent_dir + '/pyrometer.csv')
    if 'TimeStamp Adjust' in check_cols.columns:
        date_cols.append('TimeStamp Adjust')
        
    df = pd.read_csv(parent_dir + '/pyrometer.csv', parse_dates=date_cols)
    dataSearch(parent_dir, partial(splitPyrometerCsv, df))