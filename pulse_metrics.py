import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from helper_functions import selectFile

# https://doi.org/10.1179/136217199101537798
def calc_metrics(file):
    df = pd.read_csv(file)

    window = 5000   # rolling window length
    overlap = 4500  # how many points neighboring windows share
    timestep = (window - overlap) * (1 / 20e3)

    # list of indices of window starts
    starts = np.arange(0, len(df), window - overlap)

    # init arrays for metrics
    sti   = np.zeros(len(starts))
    ssi   = np.zeros(len(starts))
    ati   = np.zeros(len(starts))
    asi   = np.zeros(len(starts))
    pr    = np.zeros(len(starts))
    times = np.full(len(starts), datetime(year=1970, month=1, day=1))

    start_time = datetime.strptime(df['Timestamp'][int((window - overlap) / 2)][:-4], '%Y-%m-%d %H:%M:%S.%f')  # time of middle sample in first window

    # iterate over windows
    for i, start in enumerate(starts):

        # catch the case where the last window extends out of bounds
        if start + window > len(df):
            stop = len(df)
        else:
            stop = start + window

        # get windowed data
        voltage = df['Scaled_Voltage(V)'][start:stop]
        current = df['Scaled_Current(A)'][start:stop]

        # rough time interpolation. See above for timestep definition
        times[i] = start_time + timedelta(seconds=(timestep * i))

        # get intermediate values
        v_mean = np.mean(voltage)
        v_bk   = np.mean(voltage <= v_mean)

        i_mean = np.mean(current)
        i_bk = np.mean(current <= i_mean)

        # calculate metrics
        sti[i] = 1 - (np.min(voltage) / v_mean)
        ssi[i] = 1 - (v_bk / v_mean)

        ati[i] = 1 - (np.min(current) / i_mean)
        asi[i] = 1 - (i_bk / i_mean)

        pr[i]  = (i_bk * v_bk)/(i_mean * v_mean)

    return pd.DataFrame({'Time':times, 'STI':sti, 'SSI':ssi, 'ATI':ati, 'ASI':asi, 'PR':pr})

if __name__ == '__main__':
    lembox_file = Path(selectFile('select lembox_data.csv'))

    df = calc_metrics(lembox_file)

    df.to_csv(lembox_file.parent / 'power_supply_metrics.csv', index=False)