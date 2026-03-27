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

def get_heat_input(pos_lem, d):

    # calculate instantaneous heat input
    hi = []
    for i in range(len(pos_lem['time'])):
        if pos_lem['Vel_comb(mm/s)'][i] == 0:
            hi.append(np.inf)
        else:
            hi.append((pos_lem['Voltage(V)'][i] * pos_lem['Current(A)'][i])/ pos_lem['Vel_comb(mm/s)'][i])

    #avgLen = 10000
    #avgHI = getRollingAvg(hi, avgLen)


    t_rel = []
    startTime = pos_lem.loc[0, 'time']
    for t in pos_lem['time']:
        t_rel.append((t - startTime).seconds)


    fig, ax = plt.subplots(2,1, sharex=False, constrained_layout=True)
    ax[0].scatter(t_rel, hi)
    #ax[1].scatter(t_rel[avgLen - 1:], avgHI)
    ax[1].scatter(t_rel, pos_lem['Vel_comb(mm/s)'])
    ax[0].set_ylim(0,3000)
    #ax[1].set_ylim(0,3000)
    fig.set_size_inches(22,10)

    if not os.access(d + '/visualizations/', os.R_OK):
        os.mkdir(d + '/visualizations/')

    plt.savefig(d + '/visualizations/heat_input.png')
    
    dfAddColumn(pos_lem, hi, 'Instantaneous_Heat_Input(J/mm)')
    dfToCsv(pos_lem, d + '/aligned_data.csv')

    return pos_lem

def main():
    if len(sys.argv) != 2: dir = selectFolder()
    else: dir = sys.argv[1]

    print('Reading data...')
    df = alignData(dir)
    
    print('Computing instantaneous Heat Input...')
    get_heat_input(df, dir)


if __name__ == '__main__': main()