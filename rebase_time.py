import pandas as pd
import numpy as np
import skimage
import scipy
from pathlib import Path
from datetime import timedelta, datetime, timezone
from functools import partial
import matplotlib.pyplot as plt

import helper_functions
import batch_process

logger = helper_functions.setup_logger(__name__)

# function that defines power supply timesteps relative to the moment that the robot starts moving.
# the assumption is that the robot starts moving the moment arc initiation voltage drops
def temporal_alignment(dir, ignition_wait_time = .2, flag_plot = False):

    dir = Path(dir)

    # commanded time (in indices) that the robot idles at weld start before robot motion begins
    ignition_wait_id = int(ignition_wait_time * 250)

    # read modified data. TODO: this operation should happen in memory
    ps  = pd.read_csv(dir / 'lembox_data__clean.csv', parse_dates=['Timestamp'])
    rob = pd.read_csv(dir / 'robot_data__clean.csv', parse_dates=['SystemTime'])

    try:
        mic = pd.read_csv(dir / 'microphone_data__clean.csv', parse_dates=['Absolute Time'])
    except Exception as e:
        logger.warning('microphone_data__clean.csv not found. continuing...')
        mic = None

    # data with features we plan to align to
    data  = {'power supply': [ps['Scaled_Voltage(V)']], 'robot': [rob['RIst_X'], rob['RIst_Y']],}
    if mic is not None:
        data['microphone'] = [mic['Amplitude']]

    sr    = {'power supply': 20e3, 'robot': 250, 'microphone': 48e3}

    # start with rough interpolated times for power supply, microphone. This ensures that things are sequential for peak detection
    time  = {
        'power supply': pd.Series((ps['Timestamp'][0] + timedelta(hours=5)).timestamp() + np.arange(len(ps)) * (1/sr['power supply'])),
        'robot': np.asarray([(t + timedelta(hours=5)).timestamp() for t in rob['SystemTime']]), # to timestamp
    }
    if mic is not None:
        time['microphone'] = pd.Series((mic['Absolute Time'][0] + timedelta(hours=5)).timestamp() + np.arange(len(mic)) * (1/sr['microphone']))

    if flag_plot:
        plt.rcParams['lines.markersize'] = .1
        fig, ax = plt.subplots(4, 1, layout='constrained', sharex=True)
        i = 0

    peaks = {}
    for key, d in data.items():

        # detect rising edges in data
        peaks[key] = []
        for stream in d:

            diff   = np.diff(stream.abs())
            thresh = skimage.filters.threshold_otsu(diff) # experimentally determined to be most consistent method. triangle may be better for all but mic

            # microphone filter must be more permissive to capture start.
            # alternatively, maybe some windowed spectral method would be easier to catch edges in. Though, window adds slop to time alignment
            if key == 'microphone':
                thresh /= 10

            peaks[key].append(scipy.signal.find_peaks(diff, height=thresh, threshold=thresh)[0] + 1)

            if flag_plot:
                ax[i].scatter(time[key], stream)
                ax[i].scatter(time[key][peaks[key][-1]], stream[peaks[key][-1]], c='red', s=5)

                ax[i].set_title(key)
                i += 1

    if flag_plot: 
        # show previous plot
        fig.suptitle('Visualize all detected peaks')
        plt.show()

        # setup next plot
        fig, ax = plt.subplots(4, 1, layout='constrained', sharex=True)
        i = 0

    weld_start_id = {}
    for key, peak in peaks.items():
        
        for j, axis in enumerate(peak):

            # disregard arc initiation voltage spike. Catch first non-initiation moment
            if key == 'power supply':
                weld_start_id[key] = axis[1]

                if flag_plot:
                    ax[i].scatter(time[key], data[key][j])
                    ax[i].scatter(time[key][weld_start_id[key]], data[key][j][weld_start_id[key]], c='red', s=5)

            # robot start is the first moment in x or y position with a peak (might need to exclude pre-weld motion for generalizability?)
            elif key == 'robot':

                # remove rare false peaks in steady data
                count = 0
                while axis[count + 1] - axis[count] > 5 and count + 2 < len(axis):
                    count += 1

                if flag_plot:
                    ax[i].scatter(time[key], data[key][j])
                    ax[i].scatter(time[key][axis[count]], data[key][j][axis[count]], c='red', s=5)
                
                if key not in weld_start_id.keys() or axis[count] - ignition_wait_id < weld_start_id[key]:

                    weld_start_id[key] = axis[count] - ignition_wait_id

            else:
                weld_start_id[key] = axis[0]

                if flag_plot:
                    ax[i].scatter(time[key], data[key][j])
                    ax[i].scatter(time[key][weld_start_id[key]], data[key][j][weld_start_id[key]], c='red', s=5)

            if flag_plot:

                ax[i].set_title(key)
                i += 1

    weld_stop_id = {'power supply':peaks['power supply'][0][-1]}

    if weld_stop_id['power supply'] >= len(ps):
        weld_stop_id['power supply'] = len(ups) - 1

    if flag_plot:
        fig.suptitle('Visualize weld start samples')
        plt.show()

    weld_start_time = time['robot'][weld_start_id['robot']]

    for key, t in time.items():

        # robot time (based on IPOC) is not reinterpolated
        if key != 'robot':
            time[key] = [(timedelta(seconds=(i - weld_start_id[key]) * (1/sr[key])) + datetime.fromtimestamp(weld_start_time)) for i in range(len(t))]
    
    weld_stop_time = time['power supply'][weld_stop_id['power supply']].timestamp()

    flag_is_welding = pd.Series([t > weld_start_time and t < weld_stop_time for t in time['robot']])

    ps['Timestamp']        = time['power supply']
    rob['flag_is_welding'] = flag_is_welding

    if mic is not None:
        mic['Absolute Time'] = time['microphone']

    if flag_plot:
        plt.scatter(ps['Timestamp'], ps['Scaled_Voltage(V)'])
        plt.scatter(rob['SystemTime'], rob['flag_is_welding'])
        plt.title('test welding flag')
        plt.show()

    # show temporally aligned data
    if flag_plot:
        fig, ax = plt.subplots(4, 1, layout='constrained', sharex=True)
        ax[0].scatter(ps['Timestamp'], ps['Scaled_Voltage(V)'])
        ax[1].scatter(rob['SystemTime'], rob['RIst_X'])
        ax[2].scatter(rob['SystemTime'], rob['RIst_Y'])
        
        if mic is not None:
            ax[3].scatter(mic['Absolute Time'], mic['Amplitude'])

        plt.show()

    # save data with new timestamps
    ps .to_csv(dir / 'lembox_data__clean.csv'    , index=False)
    rob.to_csv(dir / 'robot_data__clean.csv'     , index=False)

    if mic is not None:
        mic.to_csv(dir / 'microphone_data__clean.csv', index=False)

if __name__ == '__main__':

    [dir, _] = helper_functions.setup_kwargs(__name__)

    batch_process.dataSearch(dir, partial(temporal_alignment, ignition_wait_time=.2, flag_plot=True))