from batch_process import dataSearch
from data_manipulation import selectFolder, getStartStop
from position import plotMultiBeadColor
import matplotlib.pyplot as plt
import pandas as pd
import math
import os

pos = [[], [], []]
vals = []
ids =['Current(A)', 'Voltage(V)', 'Avg_Current(A)', 'Avg_Voltage(V)']

def buildPart(e):
    avgScale=int(48000/200)

    df = pd.read_csv(e.path + '/aligned_data.csv')

    start, stop = getStartStop(df['Avg_Voltage(V)'], 1)
    df = df[start:stop]
    df = df.reset_index()

    for i in range(math.floor(len(df['time']) / avgScale) - 1):
        pos[0].append(df['Pos_x(mm)'][i*avgScale])
        pos[1].append(df['Pos_y(mm)'][i*avgScale])
        pos[2].append(df['Pos_z(mm)'][i*avgScale])

    for e, id in enumerate(ids):
        for i in range(math.floor(len(df['time']) / avgScale) - 1):
            vals[e].append(df[id][i*avgScale:(i+1)*avgScale].mean())

    return

def main():
    f = selectFolder()
    for id in ids: vals.append([])

    dataSearch(f, buildPart, 0, 1)

    parent, search = os.path.split(f)

    for i, id in enumerate(ids):
        plotMultiBeadColor(pos, vals[i], id)
        plt.savefig(parent + '/' + search + '_' + id + '.png')

    return

if __name__ == "__main__": main()