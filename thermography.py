import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
        pos.append([event.xdata, event.ydata])
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
    frameIntensity = []
    times     = []
    def getPixelIntensityTrend(e):
        frame = np.load(e.path, allow_pickle=True)
        pixelIntensity.append(frame.item()['frame'][pix[1]][pix[0]]/ (math.pow(2,16) - 1))
        frameIntensity.append(frame.item()['frame'] / (math.pow(2,16) - 1))
        times.append(frame.item()['timestamp'])
        return
    
    dataSearch(dir, getPixelIntensityTrend, False, 'FLIR-Frame')

    
    for i, t in enumerate(times):
        t_micro = float(t[-3:])/math.pow(10,3)
        t = time.mktime(time.strptime(t[:-4], '%Y-%m-%d %H:%M:%S'))
        t += t_micro

        times[i] = t

    df = pd.DataFrame(data={'timestamps':times, 'i_pix':pixelIntensity, 'i_frame':frameIntensity})
    df.sort_values(by=['timestamps'], inplace=True)

    return [df['timestamps'].to_list(), df['i_pix'].to_list(), df['i_frame'].tolist()]

def drawTimeAnimation(data, pix, dir):
    
    # modify getFrameData to pull full frames as well as selected pixel intensities. sort frames, intensities. drawNextFrame should be counting current frame count, then windowing intensity/timestamp to match framecount
    fig, ax = plt.subplots(2,1)
    step = 10

    def initFrame():
        fig.suptitle('Temperature at Point')

        ax[1].set_ylim([min(data[1]), max(data[1]) + 0.1])
        ax[1].set_xlim([min(data[0]) -10, max(data[0]) + 10])

        frame = []
        frame.append(ax[0].imshow(T_frame, vmin=0, vmax=1))
        frame.append(ax[1].scatter(t, T_pix, s=0.25, c='#036ffc'))
        return frame

    def drawNextFrame(fc, dat):

        print(fc)

        t = dat[0][:fc]
        T_pix = dat[1][:fc]
        T_frame = dat[2][fc]

        frame = []
        frame.append(ax[0].imshow(T_frame, vmin=0, vmax=1))
        frame.append(ax[1].scatter(t, T_pix, s=0.25, c='#036ffc'))
        return frame
    
    frame_count = []
    for f in range(1, len(data[0]), step):
        frame_count.append(f)

    for i, f in enumerate(frame_count):
        if ((f-1)/step) % 2 == 0:
            data[2][f][pix[1]][pix[0]] = 0
            data[2][f][pix[1] + 1][pix[0]] = 0
            data[2][f][pix[1] - 1][pix[0]] = 0
            data[2][f][pix[1]][pix[0] + 1] = 0
            data[2][f][pix[1]][pix[0] - 1] = 0

        else:
            data[2][f][pix[1]][pix[0]] = 1
            data[2][f][pix[1] + 1][pix[0]] = 1
            data[2][f][pix[1] - 1][pix[0]] = 1
            data[2][f][pix[1]][pix[0] + 1] = 1
            data[2][f][pix[1]][pix[0] - 1] = 1

    t = data[0][0]
    T_pix = data[1][0]
    T_frame = data[2][0]


    # make animation with flir frames in 1 region of subplot, updating intensity values in another region
    ani = FuncAnimation(fig, partial(drawNextFrame, dat=data), frames=frame_count, init_func=initFrame, blit=True)
    ani.save(os.path.split(dir)[0] + '/visualizations/thermal.mp4', fps=int(15/step))

    return

def main():
    dir = selectFolder()
    dir += '/FLIR'

    pixel = getPixels(dir, 1)

    print(pixel)

    df = getFrameData(pixel, dir)

    drawTimeAnimation(df, pixel, dir)

    return

if __name__ == '__main__': main()