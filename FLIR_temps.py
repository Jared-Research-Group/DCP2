import pickle
import os
import pandas as pd
import numpy as np
from thermography import getPixels, getFrameData, getTempData, selectFolder

from batch_process import dataSearch

def getTempPickles(dir, temp_type, pix=None):
    
    timestamps, i_pix, frame_paths = getFrameData(dir + '/FLIR', pix)
    dat = getTempData(pd.DataFrame({'timestamp':timestamps, 'i_pix':i_pix}), dir + '/FLIR')
    
    if not os.access(dir + '/temp_data/' + temp_type + '/', os.R_OK):
        os.mkdir(dir + '/temp_data/' + temp_type + '/')

    for index, row in dat.iterrows():
        temp = row['temp_pix']
        temp = np.array(temp)

        with open(dir + '/temp_data/' + temp_type + '/' + str(row['timestamp']).replace(':', '_') + '.pkl', 'wb') as file:
            pickle.dump(temp, file)

def recursiveTempSelection(entry):
    dir = entry.path

    if not os.access(os.path.split(dir)[0] + '/pix' + '.npy', os.R_OK):
        pix = getPixels(dir + '/FLIR', 2)
        np.save(os.path.split(dir)[0] + '/pix' + '.npy', np.array(pix))
    else:
        pix = np.load(os.path.split(dir)[0] + '/pix' + '.npy')

    if not os.access(dir + '/temp_data/', os.R_OK):
        os.mkdir(dir + '/temp_data/')

    # just RoI
    getTempPickles(dir, 'roi', pix)

    pix = []
    for i in range(464):
        pix.append([])
        for j in range(348):
            pix[-1].append([i, j])
        
    print(pix)
    # whole frame
    getTempPickles(dir, 'full', pix)

if __name__ == '__main__':
    dir = selectFolder()

    dataSearch(dir, recursiveTempSelection)

