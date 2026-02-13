from thermography import getPixels, getFrameData, drawTimeAnimation
from core_scripts.data_manipulation import selectFolder

if __name__ == '__main__':
    dir = selectFolder()
    dir += '/FLIR'

    p = getPixels(dir, 2)

    timestamps, temps, frames = getFrameData(p, dir)

    drawTimeAnimation(timestamps, temps, frames, p, dir)

    #for t in dat[1]:
    #    print(t[0][0])