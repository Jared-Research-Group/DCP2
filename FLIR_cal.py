import pandas as pd
import numpy as np

from thermography import getPixels, getFrameData, drawTimeAnimation
from data_manipulation import selectFolder

if __name__ == '__main__':
    dir = selectFolder()

    p = getPixels(dir + '/FLIR', 2)

    timestamps, temps, frames = getFrameData(dir + '/FLIR', p)

    print(temps)

    cal_data = {}
    for i, fr in enumerate(frames): cal_data[fr] = temps[i]

    df = pd.read_csv(dir + '/aligned_data.csv')

    print(df.shape)

    np.zeros((df.shape[0], len(p), len(p[1])))