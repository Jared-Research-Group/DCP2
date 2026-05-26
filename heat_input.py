import sys
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import os
import math

from data_manipulation import selectFolder, getRollingAvg, dfAddColumn, dfToCsv
from align_data import alignData

plt.rcParams['lines.markersize'] = 0.005
lem_rate = 20000 # Hz
scale = 5000           # num data points used in calculation of velocity. Small numbers produce very large velocities, or zero velocities (resolution of position data)

def get_heat_input(pos_lem, d, plot=False):

    # calculate instantaneous heat input
    #hi = []
    #for i in range(len(pos_lem['time'])):
    #    if pos_lem['Vel_comb(mm/s)'][i] == 0:
    #        hi.append(np.inf)
    #    else:
    #        hi.append((pos_lem['Voltage(V)'][i] * pos_lem['Current(A)'][i])/ pos_lem['Vel_comb(mm/s)'][i])


    #hi = pos_lem['Voltage(V)'] * pos_lem['Current(A)'] / pos_lem['Vel_comb(mm/s)']

    hi = pos_lem[pos_lem['Vel_comb(mm/s)'].isna() == False].eval('`Voltage(V)` * `Current(A)` / `Vel_comb(mm/s)`')

    avgTime = .5 # s
    avgLen = int(avgTime * lem_rate)
    avgHI = hi.rolling(window=avgLen).mean()

    NaN = pd.Series([np.nan])
    for i in range(len(pos_lem) - len(hi)):
        hi = pd.concat([NaN, hi], ignore_index=True)
        avgHI = pd.concat([NaN, avgHI], ignore_index=True)

    time = pos_lem['time']

    fig, ax = plt.subplots(2,1, sharex=True, constrained_layout=True)
    ax[0].scatter(time, hi, color='blue', label='Instantaneous')
    ax[0].scatter(time, avgHI, color='orange', label=(str(avgTime) + 's Rolling Average'))
    ax[0].set_ylabel('Heat Input (J/mm)')
    ax[0].legend(markerscale=500)

    ax[0].set_ylim(bottom=0)
    if ax[0].get_ylim()[1] > 3000:
        ax[0].set_ylim(top=3000)

    ax[1].scatter(time, pos_lem['Vel_comb(mm/s)'])
    ax[1].set_ylabel('Torch Velocity (mm/s)')
    ax[1].set_xlabel('Time ')

    if ax[1].get_ylim()[1] > 50:
        ax[1].set_ylim(top=50)

    if ax[1].get_ylim()[0] < -50:
        ax[1].set_ylim(bottom=-50)

    fig.set_size_inches(22,10)

    if not os.access(d + '/visualizations/', os.R_OK):
        os.mkdir(d + '/visualizations/')

    plt.savefig(d + '/visualizations/heat_input.png')
    
    dfAddColumn(pos_lem, hi, 'Instantaneous_Heat_Input(J/mm)')
    dfToCsv(pos_lem, d + '/aligned_data.csv')

    if plot:
        plt.show()

    return pos_lem

def main():
    if len(sys.argv) != 2: dir = selectFolder()
    else: dir = sys.argv[1]

    print('Reading data...')
    df = alignData(dir, True)
    
    print('Computing instantaneous Heat Input...')
    get_heat_input(df, dir)


if __name__ == '__main__': main()