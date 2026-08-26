import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from thermography import getPixels, getTempData_np, selectFolder
from batch_process import dataSearch
import helper_functions

# get list of frame timestamps, selected pixel intensities, paths to frames
def get_framewise_temps(dir, temp_type, pix):
    dir = Path(dir)

    if not os.access(dir.parent / ('temp_data_' + dir.parent.name) / temp_type, os.R_OK):
        os.mkdir(dir.parent / ('temp_data_' + dir.parent.name) / temp_type)

    model = helper_functions.get_FLIR_model(dir)

    # function called in recursive file search
    def find_frame_temp(e):
        nonlocal pix, model

        # read/store timestamp, filename 
        frame = np.load(e, allow_pickle=True)
        time = frame.item()['timestamp']
        time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S.%f')

        frame_dat = frame.item()['frame']


        pixel_intensity = np.zeros(pix.shape[:2])

        for i in range(len(pix)):
            for j in range(len(pix[i])):
                p = pix[i][j]
                pixel_intensity[i][j] = frame_dat[p[1]][p[0]]

        #raw_data = pd.DataFrame({'timestamp':time, 'i_pix':pixel_intensity})
        temp_data = getTempData_np(pixel_intensity, dir, model=model)
        #temp = temp_data['temp_pix'].to_list()

        with open(dir.parent / ('temp_data_' + dir.parent.name) / temp_type / (str(time).replace(':', '_') + '.csv'), 'w') as f:
            np.savetxt(f, temp_data, delimiter=',')
            f.close()
        
        return
    
    #print('         Reading FLIR Frames...')
    dataSearch(dir, find_frame_temp, id='FLIR-Frame', progressBar=False)
    #print('\nData saved!')


def recursiveTempSelection(entry):
    dir = Path(entry)

    if not os.access(dir.parent / 'pix.npy', os.R_OK):
        pix = getPixels(dir / 'FLIR', 2)
        np.save(dir.parent / 'pix.npy', np.array(pix))
    else:
        pix = np.load(dir.parent / 'pix.npy')

    if not os.access(dir / ('temp_data_' + dir.name), os.R_OK):
        os.mkdir(dir / ('temp_data_' + dir.name))

    # just RoI
    if not os.access(dir / ('temp_data_' + dir.name) / 'roi', os.R_OK):
        get_framewise_temps(dir / 'FLIR', 'roi', pix)

    """
    pix = []
    for i in range(464):
        pix.append([])
        for j in range(348):
            pix[-1].append([i, j])
        
    # whole frame
    if not os.access(dir / ('temp_data_' + dir.name) / 'full', os.R_OK):
        get_framewise_temps(dir / 'FLIR', 'full', pix)
    """

def getTempSequences(dir):

    dir = Path(dir)

    frames = [file for file in dir.iterdir() if file.is_file()]

    shape = (np.genfromtxt(frames[0], delimiter = ',').shape) + (len(frames),)
    arr = np.zeros(shape = shape, dtype = np.float64)

    for i, frame in tqdm(enumerate(frames)):

        arr[:, :, i] = np.genfromtxt(frame, delimiter = ',')

    np.save(dir.parent / 'sequences.npy', arr)


if __name__ == '__main__':

    dir = selectFolder()

    dir = Path(dir)

    #pix = getPixels(Path(r"C:\Users\w2w\Data\rebase\modified_data\data_collection_20251211_165904") / 'FLIR', 2)
    #np.save(dir / 'pix.npy', np.array(pix))

    #dataSearch(dir, recursiveTempSelection, progressBar = True)

    getTempSequences(dir)

