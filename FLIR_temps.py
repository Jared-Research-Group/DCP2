import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from thermography import getPixels, getTempData, selectFolder
from batch_process import dataSearch
import helper_functions

# get list of frame timestamps, selected pixel intensities, paths to frames
def get_framewise_temps(dir, temp_type, pix=None):

    dir = Path(dir)
    output_folder = dir.parent / ('temp_data_' + dir.parents[1].stem) / temp_type

    if not output_folder.is_dir():
        os.mkdir(output_folder)

    # function called in recursive file search
    def find_frame_temp(e):
        nonlocal pix

        # read/store timestamp, filename 
        frame = np.load(e, allow_pickle=True)
        time = frame.item()['timestamp']
        time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S.%f')

        frame_dat = frame.item()['frame']

        # horizontal rectangular region of pixels selected

        pixel_intensity = np.array(pix.shape)

        for i in tqdm(range(len(pix))):
            for j in range(len(pix[i])):
                p = pix[i][j]
                pixel_intensity[i][j].append(frame_dat[p[1]][p[0]])

        raw_data = pd.DataFrame({'timestamp':time, 'i_pix':pixel_intensity})
        temp_data = getTempData(raw_data, dir)
        temp = np.array(temp_data['temp_pix'].to_list())

        np.savetxt(output_folder / (str(time).replace(':', '_') + '.csv'), temp, delimiter=',')
        
        return
    
    #print('         Reading FLIR Frames...')
    dataSearch(dir, find_frame_temp, id='FLIR-Frame', progressBar=False)
    #print('\nData saved!')


def recursiveTempSelection(entry):
    dir = Path(entry)

    if not (dir.parent / ('pix.npy')).is_file():
        pix = getPixels(dir + '/FLIR', 2)
        np.save(dir.parent / ('pix.npy'), np.array(pix))
    else:
        pix = np.load(dir.parent / 'pix.npy')

    if not os.access(dir / ('temp_data_' + dir.stem), os.R_OK):
        os.mkdir(dir / ('temp_data_' + dir.stem))

    # just RoI
    if not os.access(dir / ('temp_data_' + dir.stem) / 'roi', os.R_OK):
        get_framewise_temps(dir / 'FLIR', 'roi', pix)

    """
    pix = []
    for i in range(464):
        pix.append([])
        for j in range(348):
            pix[-1].append([i, j])
        
    # whole frame
    if not os.access(dir / ('temp_data_' + dir.stem) / 'full', os.R_OK):
        get_framewise_temps(dir / 'FLIR', 'full', pix)
        """

if __name__ == '__main__':

    dir = selectFolder()

    dir = Path(dir)

    #pix = getPixels(Path(r"C:\Users\w2w\Data\rebase\modified_data\data_collection_20251211_165904") / 'FLIR', 2)
    #np.save(dir / 'pix.npy', np.array(pix))

    dataSearch(dir, recursiveTempSelection)

