import os

from xiris import getFrameData, getFrameList
from data_manipulation import selectFolder

if __name__ == '__main__':
    dir = selectFolder()

    if os.path.split(dir)[1] != 'raw': 
        if os.path.split(dir)[1] == 'Xiris':
            dir += '/raw'
        else:
            dir += '/Xiris/raw'

    fr_names = getFrameList(dir)
    frames = getFrameData(dir[:-10], fr_names)