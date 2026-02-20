# this file contains helper functions to perform computations and store calcualted values

import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd
import sys
import math
import tkinter as tk
from tkinter import filedialog
import os

import cython

def getRollingAvg(arr, avgLen=1000):
    if np.isnan(arr).any():
        print('Array has NaN')

    if avgLen <= 1:
        print('Average window too small: must be larger than 2')
        sys.exit(1)

    if type(arr) != list: arr = arr.tolist()

    cdef list avg = []
    for i in range(avgLen-1):
        avg.append(float('NaN'))

    cdef list buffer = arr[:avgLen - 1]
    
    for r in range(len(buffer)):
        if math.isnan(buffer[r]): buffer[r] = buffer[r-1]
    
    cdef double s = sum(buffer)
    cdef int v

    for v in range(len(arr[avgLen-1:])):
        v = v+avgLen - 1

        buffer.append(arr[v])
        if math.isnan(buffer[-1]):
            buffer[-1] = buffer[-2]
        s += buffer[-1]

        avg.append(s/avgLen)    
        s -= buffer.pop(0)

    return avg

def getRollingStdDev(arr, int sd_scale=5000):

    cdef list avg_arr = getRollingAvg(arr, sd_scale)
    avg_arr = avg_arr[sd_scale - 1:]

    cdef list buffer = [[], []]

    for a in arr[:sd_scale - 1]:
        buffer[0].append(math.pow(a,2))
        buffer[1].append(a)

    cdef list s = []
    s.append(sum(buffer[0]))
    s.append(sum(buffer[1]))

    cdef list sd = []
    for i, a in enumerate(arr[sd_scale - 1:]):
        buffer[0].append(math.pow(a,2))
        buffer[1].append(a)

        s[0] += buffer[0][-1]
        s[1] += buffer[1][-1]

        sd.append(math.sqrt((s[0] - (2 * avg_arr[i] * s[1]))/sd_scale + math.pow(avg_arr[i],2)))

        s[0] -= buffer[0].pop(0)
        s[1] -= buffer[1].pop(0)

    return sd

def getRollingSkew(arr, int sk_scale, list sd):

    cdef list avg_arr = getRollingAvg(arr, sk_scale)
    avg_arr = avg_arr[sk_scale - 1:]

    if not sd:
        sd = getRollingStdDev(arr, sk_scale)

    cdef list buffer = [[], [], []]
    for a in arr[:sk_scale - 1]:
        buffer[0].append(math.pow(a,3))
        buffer[1].append(math.pow(a,2))
        buffer[2].append(a)

    cdef list s = []
    s.append(sum(buffer[0]))
    s.append(sum(buffer[1]))
    s.append(sum(buffer[2]))

    cdef list sk = []
    for i, a in enumerate(arr[sk_scale - 1:]):
        buffer[0].append(math.pow(a,3))
        buffer[1].append(math.pow(a,2))
        buffer[2].append(a)

        s[0] += buffer[0][-1]
        s[1] += buffer[1][-1]
        s[2] += buffer[2][-1]

        sk.append((((s[0] - (3 * avg_arr[i] * s[1]) + (3 * math.pow(avg_arr[i], 2) * s[2]))/sk_scale) - math.pow(avg_arr[i], 3)) / math.pow(sd[i], 3))

        s[0] -= buffer[0].pop(0)
        s[1] -= buffer[1].pop(0)
        s[2] -= buffer[2].pop(0)

    return sk

def getRollingKurtosis(arr, int k_scale, list sd):

    cdef list avg_arr = getRollingAvg(arr, k_scale)
    avg_arr = avg_arr[k_scale - 1:]

    if not sd:
        sd = getRollingStdDev(arr, k_scale)

    cdef list buffer = [[], [], [], []]
    for a in arr[:k_scale - 1]:
        buffer[0].append(math.pow(a,4))
        buffer[1].append(math.pow(a,3))
        buffer[2].append(math.pow(a,2))
        buffer[3].append(a)

    cdef list s = []
    s.append(sum(buffer[0]))
    s.append(sum(buffer[1]))
    s.append(sum(buffer[2]))    
    s.append(sum(buffer[3]))

    cdef list k = []
    for i, a in enumerate(arr[k_scale - 1:]):
        buffer[0].append(math.pow(a,4))
        buffer[1].append(math.pow(a,3))
        buffer[2].append(math.pow(a,2))
        buffer[3].append(a)

        s[0] += buffer[0][-1]
        s[1] += buffer[1][-1]
        s[2] += buffer[2][-1]
        s[3] += buffer[3][-1]

        k.append((((s[0] - (4 * avg_arr[i] * s[1]) + (6 * math.pow(avg_arr[i], 2) * s[2]) - (4 * math.pow(avg_arr[i], 3) * s[3]))/k_scale) + math.pow(avg_arr[i], 4)) / math.pow(sd[i], 4))

        s[0] -= buffer[0].pop(0)
        s[1] -= buffer[1].pop(0)
        s[2] -= buffer[2].pop(0)
        s[3] -= buffer[3].pop(0)


    return k

# takes an array and a limit value and returns the start:stop indices that bound  (array's value) > testLimit
def getStartStop(testVal, testLimit = 1):

    startTime = 0
    for t, v in enumerate(testVal):
        if v > testLimit: 
            startTime = t
            break

    stopTime = 0
    for t, v in enumerate(testVal[1:]):                                     # changed this to [1:], keep an eye on this if issues arise
        if t < startTime:
            continue

        if testVal[t - 1] >=  testLimit and v < testLimit:
            stopTime = t

    if stopTime == 0: stopTime = len(testVal)

    return startTime, stopTime

def quickPlot(data, s=0.005):
    plt.style.use('_mpl-gallery')
    fig, ax = plt.subplots(len(data), 1, constrained_layout=True)

    if len(data) > 1:
        for j, d in enumerate(data):
            for i, val in enumerate(d):
                ax[i][j].scatter(val[0], val[1], s=s)
    else:
        for i, val in enumerate(data[0]):
            ax[i].scatter(val[0], val[1], s=s)

    return

def getStdDev(arr):
    return np.std(arr)

def dfAddColumn(df, arr, id):

    df[id] = arr
    return df

def dfToCsv(df, f):
    df.to_csv(f, index=False)

def dfHasColumn(df, id):

    if id in df.columns: return True
    return False

def csvHasColumn(f, id):

    if id in pd.read_csv(f, nrows=1): return True
    return False

def selectFolder():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory(
        title='Select Top-Level Folder',
        initialdir = os.path.expanduser('~')
    )

    return path

def printProgressBar(index, len, ticks=100):
        if ticks > len: ticks = int(len/10)
        
        if index % int(len/ticks) == 0:
            prog_str = '\r['
            for prog in range(int(index / int(len/ticks)) + 1):
                prog_str += '='
            for prog in range(int(ticks - (index / int(len/ticks)) - 1)):
                prog_str += ' '
            prog_str += ']'
            sys.stdout.write(prog_str)