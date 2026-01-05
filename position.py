import sys
import math
from tkinter import filedialog
import pandas as pd
import numpy as np
from data_manipulation import dfHasColumn, dfAddColumn, dfToCsv
import matplotlib.pyplot as plt

plt.style.use('_mpl-gallery')
#plt.rcParams['figure.tight_layout.use'] = True

def readRSI(f, window = 1000, forceDataUpdate = False):
    print('         Reading RSI data...')

    df = pd.read_csv(f)

    realStart = 0
    for i, t in enumerate(df['RelativeTime'][1:], start=1):
        if df['RelativeTime'][i] - df['RelativeTime'][i - 1] > 1: realStart = i

    if realStart != 0: 
        df.drop(index=range(realStart), inplace=True)
        df.reset_index(inplace=True)

    RSI_rate = int(len(df['RelativeTime'])/(df['RelativeTime'][df['RelativeTime'].index.tolist()[-1]] - df['RelativeTime'][df['RelativeTime'].index.tolist()[0]]))
    window = int(0.25*RSI_rate)

    pos_x = df['RIst_X']
    pos_y = df['RIst_Y']
    pos_z = df['RIst_Z']

    if not dfHasColumn(df, 'vel_x') or forceDataUpdate:
        
        ti=  []
        t = df['RelativeTime'][0]
        r = 1/RSI_rate
        for c in range(len(pos_x)):
            ti.append(t)
            t += r
            
        vel_x, vel_y, vel_z, vel_mag = getVelocities(pos_x, pos_y, pos_z, ti, window)
        dfAddColumn(df, vel_x, 'vel_x')
        dfAddColumn(df, vel_y, 'vel_y')
        dfAddColumn(df, vel_z, 'vel_z')
        dfAddColumn(df, vel_mag, 'vel_mag')

        #print(ti[-1])
        dfAddColumn(df, ti, 'Interpolated_Time(s)')
        dfToCsv(df, f)

    else:
        vel_x = df['vel_x']
        vel_y = df['vel_y']
        vel_z = df['vel_z']
        vel_mag = df['vel_mag']
        ti = df['Interpolated_Time(s)']

    return (pos_x, pos_y, pos_z), (vel_x, vel_y, vel_z, vel_mag), ti, RSI_rate

def getVelocities(x, y, z, t, w):
    vx = []
    vy = []
    vz = []
    vm = []

    for i in range(w):
        vx.append(float('NaN'))
        vy.append(float('NaN'))
        vz.append(float('NaN'))
        vm.append(float('NaN'))

    for p in range(len(x[w:])):
        vx.append((x[p+w] - x[p])/(t[p+w] - t[p]))
        vy.append((y[p+w] - y[p])/(t[p+w] - t[p]))
        vz.append((z[p+w] - z[p])/(t[p+w] - t[p]))
        vm.append(math.sqrt(math.pow(vx[-1], 2) + math.pow(vy[-1], 2) + math.pow(vz[-1], 2)))

    return vx, vy, vz, vm

def plotPos(pos):
    pos_x = pos[0]
    pos_y = pos[1]
    pos_z = pos[2]

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.plot(pos_x, pos_y, pos_z, ms=0.005)
    ax.set_aspect('equal')
    ax.disable_mouse_rotation()
    fig.set_size_inches(15,10)

    plt.show()

#needs conversion to **kwargs
def plotPosValColormap(pos, val, val_id='', title='', cmin = 0, cmax = -1, sparsity = 100):
    print('         Plotting position-wise colormap')

    pos_x = pos[0].to_numpy()
    pos_y = pos[1].to_numpy()
    pos_z = pos[2].to_numpy()
    val   = val.to_numpy()

    if cmax == -1: cmax = np.nanmax(val)

    pos_sparse = [[], [], []]
    val_sparse = []

    for i in range(len(pos_x)):
        if i % sparsity == 0:
            pos_sparse[0].append(pos_x[i])
            pos_sparse[1].append(pos_y[i])
            pos_sparse[2].append(pos_z[i])
            val_sparse.append(val[i])

    px = 1/plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(1920*px,1080*px))
    scatter = ax.scatter(pos_sparse[0], pos_sparse[1], pos_sparse[2], c=val_sparse, cmap='inferno', vmin=cmin, vmax=cmax, s=((14/25.4)*72) ** 2)
    scales = [np.nanmax(pos_sparse[0]) - np.nanmin(pos_sparse[0]), np.nanmax(pos_sparse[1]) - np.nanmin(pos_sparse[1]), np.nanmax(pos_sparse[2]) - np.nanmin(pos_sparse[2])]
    for i, s in enumerate(scales):
        if abs(s) < 0.1: scales[i] = 0.1

    ax.set_box_aspect(aspect=(scales[0], scales[1], scales[2]))
    ax.disable_mouse_rotation()

    ax.set_xlabel('X (mm)', labelpad=25)
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)', labelpad=-1)

    ax.locator_params(axis='y', nbins = 5)
    ax.locator_params(axis='z', nbins = 5)
    ax.locator_params(axis='z', nbins = 3)
    ax.set_title(title, y=1, loc='center')
    fig.set_size_inches(18,10)
    #ax.set_zticklabels([])
    #ax.margins(0.01)
    fig.colorbar(scatter, label=val_id, shrink=0.5)
    #fig.tight_layout()

def plotMultiBeadColor(pos, val, id, pt_sz = ((14/25.4)*72) ** 2):

    '''
    min = np.nanmin(val)
    for i, v in enumerate(val):
        val[i] = v - min

    max = np.nanmax(val)
    for i, v in enumerate(val):
        val[i] = v/max

    '''

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    #for p0, p1, p2, v in zip(pos[0], pos[1], pos[2], val):
        #ax.plot(p0,p1,p2, c=(v,0,0))

    scatter = ax.scatter(pos[0], pos[1], pos[2], c=val, cmap='inferno', s=pt_sz)

    ax.set_box_aspect(aspect=(np.nanmax(pos[0]) - np.nanmin(pos[0]), np.nanmax(pos[1]) - np.nanmin(pos[1]), np.nanmax(pos[2]) - np.nanmin(pos[2])))
    ax.disable_mouse_rotation()

    ax.set_xlabel('X (mm)', labelpad=25)
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    fig.colorbar(scatter, label=id, shrink=0.5)
    fig.set_size_inches(18,10)

    return

def main():

    if len(sys.argv) != 2:
        input_file = filedialog.askopenfilename()
    else:
        input_file = sys.argv[1]

    pos, vel = readRSI(input_file, True)
    plotPos(pos)

if __name__ == '__main__': main()
