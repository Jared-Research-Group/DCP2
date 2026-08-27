import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
import numpy as np

from data_manipulation import selectFolder
from batch_process import dataSearch

def getLayerData(dir):
    data = np.array([])

    def getTempData(f):
        nonlocal data

        dat = np.loadtxt(f.path, delimiter=',')
        data = np.append(data, dat)

    dataSearch(dir, getTempData, id='.csv', id_atFront=False)
    return data

if __name__ == '__main__':
    parent = selectFolder()

    temps = getLayerData(parent)


    y = np.arange(0, 464)
    x = np.arange(0, 348)

    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})

    def update(i):
        ax.plot_surface(X, Y, temps[i], cmap='inferno')

    ani = animation.FuncAnimation(fig=fig, func=update, frames=range(len(temps)), interval=33)
    plt.show()