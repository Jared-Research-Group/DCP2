import sys
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import math

from core_scripts.data_manipulation import selectFolder, getRollingAvg, dfAddColumn, dfToCsv
from synchronized_data import alignData

plt.rcParams['lines.markersize'] = 0.005
lem_rate = 20000 # Hz
scale = 5000           # num data points used in calculation of velocity. Small numbers produce very large velocities, or zero velocities (resolution of position data)

def get_heat_input(pos_lem, d):

    # calculate instantaneous heat input
    hi = []
    for i in range(len(pos_lem['time'])):
        if pos_lem['Vel_Comb(mm/s)'][i] == 0:
            hi.append(pos_lem['Voltage(V)'][i] * pos_lem['Current(A)'][i])
        else:
            hi.append((pos_lem['Voltage(V)'][i] * pos_lem['Current(A)'][i])/ pos_lem['Vel_Comb(mm/s)'][i])

    avgLen = 10000
    avgHI = getRollingAvg(hi, avgLen)


    fig, ax = plt.subplots(3,1, sharex=False, constrained_layout=True)
    ax[0].scatter(pos_lem['time'], hi)
    ax[1].scatter(pos_lem['time'][avgLen - 1:], avgHI)
    ax[2].scatter(pos_lem['time'], pos_lem['Vel_Comb(mm/s)'])
    ax[0].set_ylim(0,3000)
    ax[1].set_ylim(0,3000)
    fig.set_size_inches(22,10)
    plt.savefig(d + '/visualizations/heat_input.png')
    
    dfAddColumn(pos_lem, hi, 'Instantaneous_Heat_Input(J/mm)')
    dfToCsv(pos_lem, d + '/aligned_data.csv')

    return pos_lem

def main():
    if len(sys.argv) != 2: dir = selectFolder()
    else: dir = sys.argv[1]

    print('Reading data...')
    df = alignData(dir, True)
    
    print('Computing instantaneous Heat Input...')
    get_heat_input(df, dir)


if __name__ == '__main__': main()