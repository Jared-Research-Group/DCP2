from core_scripts.lembox import getLemboxData, getTimeData
import matplotlib.pyplot as plt
import sys
import os
import subprocess
from pathlib import Path
from tkinter import filedialog
import numpy as np

def selectParent():
    if len(sys.argv) == 2:
        folder_path = sys.argv[1]
    else:
        print("Please select the parent folder...")
        folder_path = filedialog.askdirectory()
    
    if not folder_path:
        print("No folder selected. Exiting...")
        sys.exit(1)
    
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory")
        sys.exit(1)

    return folder_path

def searchDir(dir):
    data_folders = []

    with os.scandir(dir) as it:
        for entry in it:
            if entry.is_dir():
                if entry.name.startswith('data_collection_'):
                    data_folders.append(dir + '/' + entry.name)
                else:
                    subsearch = searchDir(entry.path)
                    if subsearch:
                        for dir in subsearch:
                            data_folders.append(dir)

    data_folders.sort()
    return data_folders

def getStdDevs(data):
    stdDevs = [[[], []], [[], []]]

    for dir in data:
        print(dir)
        #subprocess.run([sys.executable, str(Path(__file__).parent) + '/lembox_scaling.py',
        #                                 dir + '/lembox_data.csv'], check=True)
        t, i, v, iAvg, vAvg = getLemboxData(dir + '/lembox_data.csv')

        startTime, stopTime = getTimeData(vAvg, t, dir)

        t = t[startTime:stopTime]
        i = i[startTime:stopTime]
        v = v[startTime:stopTime]
        iAvg = iAvg[startTime:stopTime]
        vAvg = vAvg[startTime:stopTime]

        #sd_i, sd_v = drawStdDev(t, i, v, "", False)
        #sd_i_avg, sd_v_avg = drawStdDev(t, iAvg, vAvg, "", False)

        stdDevs[0][0].append(np.nanstd(sd_i))
        stdDevs[0][0].append(np.nanstd(sd_v))

        stdDevs[0][0].append(np.nanstd(sd_i_avg))
        stdDevs[0][0].append(np.nanstd(sd_v_avg))

    return stdDevs

def plotStdDevs(stdDevs):
    n = []
    colors = []

    for i, dev in enumerate(stdDevs[0][0]):
        n.append(i)

    for i in range(int(len(n)/5)):
        colors.append('blue')
        colors.append('green')
        colors.append('green')
        colors.append('purple')
        colors.append('purple')

    colors.insert(69, 'red')
    
    plotDevs = [[[], []], [[], []]]
    nPlot =[]
    cPlot = []
    
    for i, sd in enumerate(stdDevs[0][0]):
        if (i + 5) % 5 != 0 and i < 68:
            plotDevs[0][0].append(stdDevs[0][0][i])
            plotDevs[0][1].append(stdDevs[0][1][i])
            plotDevs[1][0].append(stdDevs[1][0][i])
            plotDevs[1][1].append(stdDevs[1][1][i])
            
            nPlot.append(n[i])
            cPlot.append(colors[i])
            
        elif (i + 5 - 1) % 5 != 0 and i >= 68 and i < len(stdDevs[0][0]):
            plotDevs[0][0].append(stdDevs[0][0][i])
            plotDevs[0][1].append(stdDevs[0][1][i])
            plotDevs[1][0].append(stdDevs[1][0][i])
            plotDevs[1][1].append(stdDevs[1][1][i])
            
            nPlot.append(n[i])
            cPlot.append(colors[i])
            
        else:
            print(i)

    fig, ax = plt.subplots(2,2, sharex=True, constrained_layout=True)

    ax[0][0].scatter(nPlot, plotDevs[0][0], c=cPlot)
    ax[0][0].set_ylabel('Standard Deviation of Current (A)')
    ax[1][0].scatter(nPlot, plotDevs[1][0], c=cPlot)
    ax[1][0].set_ylabel('Standard Deviation of Voltage (V)')
    ax[1][0].set_xlabel('Bead Number')

    ax[0][1].scatter(nPlot, plotDevs[0][1], c=cPlot)
    ax[0][1].set_ylabel('Standard Deviation of Average Current (A)')
    ax[1][1].scatter(nPlot, plotDevs[1][1], c=cPlot)
    ax[1][1].set_ylabel('Standard Deviation of Average Voltage (V)')
    ax[1][1].set_xlabel('Bead Number')
    fig.set_size_inches(30,10)  

    '''
    f = []
    a = []

    for i in range(5):
        fig, ax = plt.subplots(2,2, sharex=True, constrained_layout=True)

        i_plot = []
        i_avg_plot = []
        v_plot = []
        v_avg_plot = []
        count = []

        for index, sd in enumerate(stdDevs[0][0]):
            if (index + 5 - i) % 5 == 0:
                i_plot.append(sd)

        for index, sd in enumerate(stdDevs[0][1]):
            if (index + 5 - i) % 5 == 0:
                i_avg_plot.append(sd)

        for index, sd in enumerate(stdDevs[1][0]):
            if (index + 5 - i) % 5 == 0:
                v_plot.append(sd)

        for index, sd in enumerate(stdDevs[1][1]):
            if (index + 5 - i) % 5 == 0:
                v_avg_plot.append(sd)

        for index, c in enumerate(n):
            if (index + 5 - i) % 5 == 0:
                count.append(c + 1)

        ax[0][0].scatter(count, i_plot)
        ax[0][0].set_ylabel('SD I')
        ax[1][0].scatter(count, v_plot)
        ax[1][0].set_ylabel('SD V')

        ax[0][1].scatter(count, i_avg_plot)
        ax[0][1].set_ylabel('SD I avg')
        ax[1][1].scatter(count, v_avg_plot)
        ax[1][1].set_ylabel('SD V avg')
        fig.set_size_inches(30,10)

        f.append(fig)
        a.append(ax)
    '''

    plt.show()

    return

def main():

    topDir = selectParent()
    dataFolders = searchDir(topDir)

    stdDevs = getStdDevs(dataFolders)
    plotStdDevs(stdDevs)

if __name__ == '__main__': main()

