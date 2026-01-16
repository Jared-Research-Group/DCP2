import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors
import matplotlib.cm     as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from functools import partial
import numpy as np
import pandas as pd
import os
import math
import time

from data_manipulation import selectFolder
from batch_process import dataSearch

def getHotFrame(dir):

    l = os.listdir(dir)
    l.sort()

    hotFrame = l[int(len(l)/2)]
    hotFrame = dir + '/' + hotFrame

    hotFrame = np.load(hotFrame, allow_pickle=True)

    return hotFrame.item()['frame']

def getPixels(dir, numPts=1):

    frame = getHotFrame(dir)

    pos = []
    def onclick(event):
        pos.append([int(event.xdata), int(event.ydata)])
        return

    fig = plt.imshow(frame)
    fig = fig.axes.figure
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

    fig.canvas.mpl_disconnect(cid)

    if len(pos) < numPts: return

    return pos[-1 * numPts:]

def getFrameData(pix, dir):

    pixelIntensity = []
    framePaths = []
    times     = []
    def getPixelIntensityTrend(e):
        frame = np.load(e.path, allow_pickle=True)
        pixelIntensity.append(frame.item()['frame'][pix[1]][pix[0]]/ (math.pow(2,16) - 1))
        framePaths.append(e.path)
        times.append(frame.item()['timestamp'])
        return
    
    dataSearch(dir, getPixelIntensityTrend, False, 'FLIR-Frame')

    startTime = time.mktime(time.strptime(times[0][:-4], '%Y-%m-%d %H:%M:%S'))
    startTime += float(times[0][-3:])/math.pow(10,3)

    for i, t in enumerate(times):
        t_micro = float(t[-3:])/math.pow(10,3)
        t = time.mktime(time.strptime(t[:-4], '%Y-%m-%d %H:%M:%S'))
        t += t_micro

        if t < startTime: startTime = t

        times[i] = t

    for i, t in enumerate(times):
        times[i] = t - startTime

    df = pd.DataFrame(data={'timestamps':times, 'i_pix':pixelIntensity, 'frame_paths':framePaths})
    df.sort_values(by=['timestamps'], inplace=True)

    return [df['timestamps'].to_list(), df['i_pix'].to_list(), df['frame_paths'].to_list()]

def drawTimeAnimation(data, pix, dir):
    
    # modify getFrameData to pull full frames as well as selected pixel intensities. sort frames, intensities. drawNextFrame should be counting current frame count, then windowing intensity/timestamp to match framecount
    fig, ax = plt.subplots(1, 2, layout='constrained')
    ax[1].set_xlim([data[0][0], data[0][-1]])
    ax[1].set_ylim([0, 1.05])
    ax[0].axis('off')

    axins = inset_axes(ax[0], width="5%", height="50%", loc='upper right')
    #axins.yaxis.set_label_position('left')
    fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=1), cmap='viridis'), cax=axins, ticks=[0, 1], ticklocation='left')

    asp = (np.diff(ax[1].get_xlim())[0] / np.diff(ax[1].get_ylim())[0]) * (348/464)
    ax[1].set_aspect(asp)
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Scaled Pixel Intensity')

    step = 500

    def drawNextFrame(fc, dat):

        print(fc)

        t = dat[0][:fc]
        T_pix = dat[1][:fc]
        T_frame = np.load(dat[2][fc], allow_pickle=True)

        T_frame = T_frame.item()['frame'] / float(math.pow(2,16) - 1)

        if ((fc-1)/step) % 2 == 0:
            T_frame[pix[1]][pix[0]] = 0
            T_frame[pix[1] + 1][pix[0]] = 0
            T_frame[pix[1] - 1][pix[0]] = 0
            T_frame[pix[1]][pix[0] + 1] = 0
            T_frame[pix[1]][pix[0] - 1] = 0

        #else:
        if True:
            T_frame[pix[1]][pix[0]] = 1
            T_frame[pix[1] + 1][pix[0]] = 1
            T_frame[pix[1] - 1][pix[0]] = 1
            T_frame[pix[1]][pix[0] + 1] = 1
            T_frame[pix[1]][pix[0] - 1] = 1

            T_frame[pix[1] + 2][pix[0]] = 1
            T_frame[pix[1] - 2][pix[0]] = 1
            T_frame[pix[1]][pix[0] + 2] = 1
            T_frame[pix[1]][pix[0] - 2] = 1

            T_frame[pix[1] + 1][pix[0] + 1] = 1
            T_frame[pix[1] + 1][pix[0] - 1] = 1
            T_frame[pix[1] - 1][pix[0] + 1] = 1
            T_frame[pix[1] - 1][pix[0] - 1] = 1

        ax[0].imshow(T_frame, vmin=0, vmax=1, cmap='viridis')
        ax[1].scatter(t, T_pix, s=0.000005, c='#036ffc')
        return
    
    frame_count = []
    for f in range(1, len(data[0]), step):
        frame_count.append(f)

    # make animation with flir frames in 1 region of subplot, updating intensity values in another region
    ani = FuncAnimation(fig, partial(drawNextFrame, dat=data), frames=frame_count)
    ani.save(os.path.split(dir)[0] + '/visualizations/thermal.mp4', fps=10)

    return

def drawTimeSeriesPixelScatter(data, pix, dir):
    fig, ax = plt.subplots(layout='constrained')
    ax.set_xlim([data[0][0], data[0][-1]])
    ax.set_ylim([0, 1.05])

    ax.scatter(data[0], data[1])
    plt.savefig(os.path.split(dir)[0] + '/visualizations/pixel.png')

def main():
    dir = selectFolder()
    dir += '/FLIR'

    pixel = getPixels(dir, 1)

    pixel = pixel[0]

    df = getFrameData(pixel, dir)

    #drawTimeAnimation(df, pixel, dir)
    drawTimeSeriesPixelScatter(df, pixel, dir)

    return

if __name__ == '__main__': main()