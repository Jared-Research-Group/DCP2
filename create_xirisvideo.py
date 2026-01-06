import numpy as np
import math
import os
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm     as cm
from matplotlib.animation import FuncAnimation
from tkinter import filedialog

from batch_process import dataSearch
from data_manipulation import selectFolder, printProgressBar

def getFrameList(dir):
    print('Collecting Frames...')

    def appendFrame(frame):
        return

    frames = dataSearch(dir, appendFrame, False, '.raw', False)
    frames.sort()

    return frames

def convertFrame(f):
    
    frame = open(f, "rb")
    frame.read(16)
    
    width = int.from_bytes(frame.read(4), 'little')
    height = int.from_bytes(frame.read(4), 'little')
    
    frame.read(32)
    
    min = math.pow(2,16)
    max = 0
    intensity = np.zeros((height, width), dtype=np.uint16)
    temps     = np.zeros((height, width), dtype=np.float64)
    for x in range(height):
        for y in range(width):
            intensity[x][y] = int(int.from_bytes(frame.read(2), 'little'))
            temps[x][y]     = ((intensity[x][y]/(math.pow(2,16) - 1)) * (1800 - 350)) + 350
            #if intensity[x][y] < min: min = intensity[x][y]
            #if intensity[x][y] > max: max = intensity[x][y]
    
    
    #scale = max - min
    #for x in range(height):
    #    for y in range(width):
    #        intensity[x][y] = int(((intensity[x][y] - min) / scale) * (math.pow(2,16) - 1))

    #cv.imshow('intensity', intensity)
    #cv.waitKey(0)

    #intensity = intensity/256
    #return intensity.astype(np.uint8)

    return intensity

def getFrameData(frames):
    dat = []

    for i, f in enumerate(frames):
        dat.append(convertFrame(f))
        printProgressBar(i, len(frames))

    return dat

def buildVideoCV2(dir, frames):
    print('Constructing Video...')

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(dir + '/Xiris.avi', fourcc, 24, (640, 512), False)

    print()
    for i, frame in enumerate(frames):
        out.write(frame)


    out.release()
    cv.destroyAllWindows()
    return

def buildVideoPyplot(dir, frames):
    print('\nConstructing Video...')

    fig, ax = plt.subplots(constrained_layout=True)
    ax.axis('off')

    fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=350, vmax=1800), cmap='viridis'), ax=ax)

    def buildFrame(dat):
        dat = convertFrame(dat)
        frame = ax.imshow(dat)

        return frame

    ani = FuncAnimation(fig, buildFrame, frames)
    ani.save(dir + '/Xiris.avi')

def main():
    if len(sys.argv) == 2:
        dir = sys.argv[1]
    else:
        dir = selectFolder()

    if os.path.split(dir)[1] != 'raw': 
        if os.path.split(dir)[1] == 'Xiris':
            dir += '/raw'
        else:
            dir += '/Xiris/raw'

    fr = getFrameList(dir)
    #fr = getFrameData(fr[800:1000])
    buildVideoPyplot(dir[:-10], fr)
    
    return  

if __name__ == '__main__': main()