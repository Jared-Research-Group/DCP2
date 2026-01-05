from lembox_visualization import getLemboxData, getStartStop
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tkinter import filedialog
import math
import numpy as np

nPointsPerFrame = 1000
limits = [[],[]]

fig, ax = plt.subplots()

def initHist():
    fig.suptitle('Histogram Animation')
    ax.set_xlabel('Voltage(V)')
    ax.set_ylabel('Current(A)')

    frame = ax.hist2d([], [], bins=250, norm='log')

    ax.set_xlim(limits[0][0], limits[0][1])
    ax.set_ylim(limits[1][0], limits[1][1])
    fig.set_size_inches(15,15)
    return frame


def splitFrameData(data, nFrames):
    hFrames = []
    for f in range(nFrames):
        hFrames.append([])
        hFrames[-1].append(data[0][:(f+1)*nPointsPerFrame])
        hFrames[-1].append(data[1][:(f+1)*nPointsPerFrame])

    return hFrames

def nextHistFrame(histFrameData):
    iFrame = histFrameData[0]
    vFrame = histFrameData[1]

    frame = ax.hist2d(vFrame, iFrame, bins=250, norm='log')
    ax.set_xlim(limits[0][0], limits[0][1])
    ax.set_ylim(limits[1][0], limits[1][1])

    
    return frame

def main():
    dir = filedialog.askdirectory()

    t, i, v, avgI, avgV = getLemboxData(dir + '/lembox_data.csv')

    start, stop = getStartStop(avgV)

    t = t[start:stop]
    i = i[start:stop]
    v = v[start:stop]
    
    limits[0] = [np.nanmin(v), np.nanmax(v)]
    limits[1] = [np.nanmin(i), np.nanmax(i)]

    nFrames = math.ceil(len(t)/nPointsPerFrame)

    hData = splitFrameData((i, v), nFrames)

    ani = FuncAnimation(fig, nextHistFrame, frames=hData, init_func=initHist)

    print('Plotting...')
    ani.save(dir + '/visualizations/histogram_animation.mp4')

    return

if __name__ == '__main__': main()