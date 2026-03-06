import numpy as np
import math
import os
import sys
import cv2 as cv
import shutil
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm     as cm
from matplotlib.animation import FuncAnimation
from functools import partial

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
    """Efficiently read raw frame data using NumPy bulk read instead of pixel-by-pixel loop"""

    cdef bytearray pixel_data
    cdef unsigned short[:,:] intensity
    with open(f, "rb") as frame:
        frame.read(16)  # Skip header
        width = int.from_bytes(frame.read(4), 'little')
        height = int.from_bytes(frame.read(4), 'little')
        frame.read(32)  # Skip remaining header
        
        # Bulk read all pixels at once instead of looping
        pixel_data = bytearray(frame.read(height * width * 2))
        intensity = np.frombuffer(pixel_data, dtype=np.uint16).reshape((height, width))
    
    return intensity

def buildSingleFrame(f, fig=None, ax=None):
    """Reuse figure/axes if provided to avoid expensive matplotlib object creation"""
    if fig is None or ax is None:
        fig, ax = plt.subplots(constrained_layout=True)
        ax.axis('off')

        #fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=350, vmax=1800), cmap='PuRd'), ax=ax)
    
    img = ax.imshow(convertFrame(f), cmap='PuRd')
    return fig, ax

def getFrameData(dir, frames):
    dir = dir + '/Xiris_Frames'
    
    mustMakeFrames = False
    if not os.access(dir, os.F_OK):
        os.mkdir(dir)
        mustMakeFrames = True
        
    pngList = []
    fig, ax = None, None
    
    for i, f in enumerate(frames):
        filename = dir + '/' + os.path.split(f)[1][:-4] + '.png'
        pngList.append(filename)
        
        if mustMakeFrames:
            # Reuse fig/ax to avoid expensive recreation
            fig, ax = buildSingleFrame(f, fig, ax)
            plt.savefig(filename, dpi=80, bbox_inches='tight')
            # Clear axes instead of closing figure
            ax.clear()
            
        printProgressBar(i, len(frames), 50)
    
    if fig is not None:
        plt.close(fig)

    return pngList, dir

def buildVideoCV2(dir, frames):
    print('\nConstructing Video...')
    
    shape = cv.imread(frames[0], flags=cv.IMREAD_COLOR).shape
    shape = (shape[1], shape[0])

    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Use more efficient codec
    output_path = dir + '/Xiris.mp4'
    out = cv.VideoWriter(output_path, fourcc, 24, shape, True)
    
    print(output_path)
    print()
    
    for i, frame in enumerate(frames):
        im = cv.imread(frame, flags=cv.IMREAD_COLOR)
        out.write(im)
        printProgressBar(i, len(frames), 50)

    out.release()
    cv.destroyAllWindows()
    return

def buildVideoPyplot(dir, frames):
    print('\nConstructing Video...')

    fig, ax = plt.subplots(constrained_layout=True)
    ax.axis('off')
    
    class imgCount:
        c=0
        
    counter=imgCount()

    #fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=350, vmax=1800), cmap='viridis'), ax=ax)

    def buildFrame(count, dat):
        dat = convertFrame(dat)
        frame = ax.imshow(dat, norm=colors.Normalize(vmin=0, vmax=math.pow(2,14)-1), cmap='viridis')
        
        printProgressBar(count.c, len(frames))
        
        count.c += 1

        return frame

    ani = FuncAnimation(fig, partial(buildFrame, counter), frames)
    ani.save(dir + '/Xiris.mp4')

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
    png_frames, frame_dir = getFrameData(dir[:-10], fr)
    buildVideoCV2(dir[:-10], png_frames)
    
    return  

if __name__ == '__main__': main()