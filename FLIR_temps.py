import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from thermography import getPixels, getTempData, selectFolder
from data_manipulation import printProgressBar
from batch_process import dataSearch

# get list of frame timestamps, selected pixel intensities, paths to frames
def get_framewise_temps(dir, temp_type, pix=None):

    if not os.access((os.path.split(dir)[0]) + '/temp_data_' + os.path.split(os.path.split(dir)[0])[1] + '/' + temp_type + '/', os.R_OK):
        os.mkdir((os.path.split(dir)[0]) + '/temp_data_' + os.path.split(os.path.split(dir)[0])[1] + '/' + temp_type + '/')

    numFrames = len([p for p in Path(dir).iterdir() if p.is_file()])        # used to print progress bar in recursive file search
    count = 0

    # function called in recursive file search
    def find_frame_temp(e):
        nonlocal pix, count, numFrames
        pixelIntensity = []

        # read/store timestamp, filename 
        frame = np.load(e.path, allow_pickle=True)
        time = frame.item()['timestamp']
        time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S.%f')

        frame_dat = frame.item()['frame']

        print(frame_dat.shape)

        # horizontal rectangular region of pixels selected

        for i in range(len(pix)):
            pixelIntensity.append([])
            for j in range(len(pix[i])):
                p = pix[i][j]
                pixelIntensity[-1].append(frame_dat[p[1]][p[0]])

        raw_data = pd.DataFrame({'timestamp':time, 'i_pix':pixelIntensity})
        temp_data = getTempData(raw_data, dir)
        temp = np.array(temp_data['temp_pix'].to_list())

        np.savetxt(os.path.split(dir)[0] + '/temp_data_' + os.path.split(os.path.split(dir)[0])[1] + '/' + temp_type + '/' + str(time).replace(':', '_') + '.csv', temp, delimiter=',')

        printProgressBar(count, numFrames)
        count += 1
        
        return
    
    print('         Reading FLIR Frames...')
    dataSearch(dir, find_frame_temp, 'FLIR-Frame')
    print('\nData saved!')


def recursiveTempSelection(entry):
    dir = entry.path

    if not os.access(os.path.split(dir)[0] + '/pix' + '.npy', os.R_OK):
        pix = getPixels(dir + '/FLIR', 2)
        np.save(os.path.split(dir)[0] + '/pix' + '.npy', np.array(pix))
    else:
        pix = np.load(os.path.split(dir)[0] + '/pix' + '.npy')

    if not os.access(dir + '/temp_data_' + os.path.split(dir)[1] + '/', os.R_OK):
        os.mkdir(dir + '/temp_data_' + os.path.split(dir)[1] + '/')

    # just RoI
    if not os.access(dir + '/temp_data_' + os.path.split(dir)[1] + '/roi', os.R_OK):
        get_framewise_temps(dir + '/FLIR', 'roi', pix)

    pix = []
    for i in range(464):
        pix.append([])
        for j in range(348):
            pix[-1].append([i, j])
        
    # whole frame
    if not os.access(dir + '/temp_data_' + os.path.split(dir)[1] + '/full', os.R_OK):
        get_framewise_temps(dir + '/FLIR', 'full', pix)

if __name__ == '__main__':
    dir = selectFolder()

    dataSearch(dir, recursiveTempSelection)

