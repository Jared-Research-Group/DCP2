# distutils: language=c++

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors
import matplotlib.cm     as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from functools import partial
import numpy as np
import pandas as pd
import os
from pathlib import Path
import math
from datetime import datetime
import cv2

from data_manipulation import selectFolder, printProgressBar, flirConversion, get_FLIR_model
from batch_process import dataSearch

import cython
from libcpp.vector cimport vector

# grabs a FLIR frame from the middle of the FLIR video. Used for allowing user selection of pixels from exemplary frame
def getHotFrame(dir):

    l = os.listdir(dir)

    hotFrame = 'FLIR-Frame-' + str(int(3*len(l)/4)) + '.npy'
    hotFrame = dir + '/' + hotFrame

    hotFrame = np.load(hotFrame, allow_pickle=True)

    return hotFrame.item()['frame']

# used to show user-selected regions
def highlight_pixel(fr, pix):
    fr[pix[1]][pix[0]] = 1
    fr[pix[1] + 1][pix[0]] = 1
    fr[pix[1] - 1][pix[0]] = 1
    fr[pix[1]][pix[0] + 1] = 1
    fr[pix[1]][pix[0] - 1] = 1

    fr[pix[1] + 2][pix[0]] = 1
    fr[pix[1] - 2][pix[0]] = 1
    fr[pix[1]][pix[0] + 2] = 1
    fr[pix[1]][pix[0] - 2] = 1

    fr[pix[1] + 1][pix[0] + 1] = 1
    fr[pix[1] + 1][pix[0] - 1] = 1
    fr[pix[1] - 1][pix[0] + 1] = 1
    fr[pix[1] - 1][pix[0] - 1] = 1
    return fr

# allows selection of an arbitrary number of pixels by hand by the user
def getPixels(dir, numPts=1):

    frame = getHotFrame(dir)/(2**16)

    pos = []
    def onclick(event):
        nonlocal fig
        nonlocal pos

        # after each click, record pixel position of click
        pos.append([int(event.xdata), int(event.ydata)])

        frame_with_highlights = frame.copy()
        highlight_pix = pos

        # if user has selected more pixels than allowed, drop the oldest pixel
        if len(highlight_pix) > numPts: highlight_pix = highlight_pix[-1 * numPts:]

        for pixel in highlight_pix:
            frame_with_highlights = highlight_pixel(frame_with_highlights, pixel)

        img.set_data(frame_with_highlights)
        fig.canvas.draw()
        fig.canvas.flush_events()

        return

    img = plt.imshow(frame, cmap='viridis')
    fig = img.axes.figure
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

    fig.canvas.mpl_disconnect(cid)

    # bad selection
    if len(pos) < numPts: return

    # if 2 pixels are given, get a matrix of all pixels in a rectangle defined at 2 corners by given pixels. Horizontal rectangular zone selection
    if numPts == 2:
        b = max([pos[-1][1], pos[-2][1]])
        t = min([pos[-1][1], pos[-2][1]])
        
        r = max([pos[-1][0], pos[-2][0]])
        l = min([pos[-1][0], pos[-2][0]])

        p = []
        for i in range(l,r + 1):
            p.append([])
            for j in range(t, b + 1):
                p[-1].append([i, j])

        pos = p
        return pos
    
    # undefined/ single pix case. Return selected pixels
    return pos[-1 * numPts:]

# get list of frame timestamps, selected pixel intensities, paths to frames
def getFrameData(dir, pix=None, printFlag=True):

    cdef list pixelIntensity = []
    cdef list framePaths     = []
    cdef list times          = []

    numFrames = len([p for p in Path(dir).iterdir() if p.is_file()])        # used to print progress bar in recursive file search

    # function called in recursive file search
    def getPixelIntensityTrend(e):
        nonlocal pix
        nonlocal pixelIntensity
        pixelIntensity.append([])

        # read/store timestamp, filename 
        frame = np.load(e.path, allow_pickle=True)
        times.append(frame.item()['timestamp'])
        framePaths.append(e.path)

        cdef unsigned short[:,:] frame_dat = frame.item()['frame']

        # variable behavior for different selections of RoI
        if pix is None: None

        # single pixel selected
        elif len(pix) == 1: pixelIntensity.append([frame_dat[pix[1]][pix[0]]/ (math.pow(2,16) - 1)])

        # horizontal rectangular region of pixels selected
        else:

            for i in range(len(pix)):
                pixelIntensity[-1].append([])
                for j in range(len(pix[i])):
                    p = pix[i][j]
                    pixelIntensity[-1][-1].append(frame_dat[p[1]][p[0]])

        if printFlag: printProgressBar(len(framePaths), numFrames)            # update user on progress
        
        return
    
    print('         Reading FLIR Frames...')
    dataSearch(dir, getPixelIntensityTrend, False, 'FLIR-Frame')
    print()

    # convert timestamp strings to datetime.datetime
    for i, t in enumerate(times):
        times[i] = datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f')

    # ensure index ascends with time
    df = pd.DataFrame(data={'timestamps':times, 'i_pix':pixelIntensity, 'frame_paths':framePaths})
    df.sort_values(by=['timestamps'], inplace=True)
    df.reset_index(inplace=True)

    timestamps = df['timestamps'].to_list()
    for i, t in enumerate(timestamps):
        timestamps[i] = t.to_pydatetime()

    if pix is not None: return timestamps, df['i_pix'].to_list(), df['frame_paths'].to_list()

    return timestamps, df['frame_paths'].to_list()
    

def getTempData(df, dir, sparsity=None):
    model = get_FLIR_model(dir)

    intensity = df['i_pix'].to_list()
    intensity = np.array(intensity)

    # only care about every [sparsity]th value. Saves memory
    if sparsity is not None:
        sparse_intensity = []

        for fr in intensity:
            for i in range(len(fr)/sparsity):
                for j in range(len(fr[0])/sparsity):
                    sparse_intensity[i][j] = intensity[i*sparsity][j*sparsity]

        intensity = sparse_intensity

    temp = flirConversion(intensity, model)

    df['temp_pix'] = pd.Series(temp.tolist())
    df.reset_index(inplace=True)

    return df

### NEEDS MODIFICATION, SHAPE OF TEMPS HAS CHANGED
def drawTimeAnimation(list times, list temps, frames, list pix, dir, int step=100):

    t_start = times[0]
    for i in range(len(times)):
        times[i] = times[i] - t_start

    # modify getFrameData to pull full frames as well as selected pixel intensities. sort frames, intensities. drawNextFrame should be counting current frame count, then windowing intensity/timestamp to match framecount
    fig, ax = plt.subplots(1, 2, layout='constrained')
    ax[1].set_xlim([times[0], times[len(times)-1]])
    ax[1].set_ylim([0, (2**16)-1])
    ax[0].axis('off')

    axins = inset_axes(ax[0], width="5%", height="50%", loc='upper right')
    axins.yaxis.set_label_position('left')
    fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=(2**16) - 1), cmap='viridis'), cax=axins, ticklocation='left')

    asp = (np.diff(ax[1].get_xlim())[0] / np.diff(ax[1].get_ylim())[0]) * (348/464)
    ax[1].set_aspect(asp)
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Scaled Pixel Intensity')

    cdef int fc_prev = 0
    def drawNextFrame(int fc):
        nonlocal fc_prev
        nonlocal times, temps, frames
        print(fc)

        cdef vector[double] t = times[fc_prev + 1:fc]
        #cdef vector[vector[vector[int]]] T_pix = temps[fc_prev + 1:fc]
        cdef vector[vector[int]] T_frame = np.load(frames[fc], allow_pickle=True).item()['frame']

        '''
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
            '''

        for i in range(len(temps)):
            for j in range(len(temps[i])):
                ax[1].scatter(t, temps[i][j][fc_prev + 1:fc], s=0.05)


        ax[0].imshow(T_frame, vmin=0, vmax=(2**16)-1, cmap='viridis')

        fc_prev = fc
        return
    
    frame_count = []
    for f in range(1, len(times), step):
        frame_count.append(f)

    # make animation with flir frames in 1 region of subplot, updating intensity values in another region
    ani = FuncAnimation(fig, drawNextFrame, frames=frame_count)
    ani.save(os.path.split(dir)[0] + '/visualizations/thermal.mp4', fps=math.ceil(30/step))

    return

def drawTimeSeriesPixelScatter(data, pix, dir):
    fig, ax = plt.subplots(layout='constrained')
    ax.set_xlim([data[0][0], data[0][-1]])
    ax.set_ylim([0, 1.05])

    ax.scatter(data[0], data[1])
    plt.savefig(os.path.split(dir)[0] + '/visualizations/pixel.png', s=0.5)

def detectEdges(frame, t1, t2):
    fr = np.load(frame, allow_pickle=True).item()['frame']
    fr = np.log1p(fr)
    fr_norm = np.uint8(cv2.normalize(fr, None, 0, 255, cv2.NORM_MINMAX))
    fr_norm = cv2.cvtColor(fr_norm, cv2.COLOR_GRAY2RGB)

    blur = cv2.GaussianBlur(fr_norm, (5,5), 1.4)
    edges = cv2.Canny(blur, threshold1=t1, threshold2=t2)

    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    combo = np.zeros(edges.shape)
    for i in range(len(fr_norm)):
        for j in range(len(fr_norm[i])):

            combo[i][j] = fr_norm[i][j]

            if np.array_equal(edges[i][j], np.array([255, 255, 255])): combo[i][j] = np.array([0, 0, 255], dtype=np.uint8)

    cv2.imshow('edges', np.uint8(combo))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    #dir = selectFolder()
    #dir += '/FLIR'

    #pixel = getPixels(dir, 1)

    #df = getFrameData(pixel, dir)
    #df = getTempData(df, pixel)

    #drawTimeAnimation(df, pixel, dir)
    #drawTimeSeriesPixelScatter(df, pixel, dir)

    return

if __name__ == '__main__': main()