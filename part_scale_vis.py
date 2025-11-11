from batch_process import dataSearch
from data_manipulation import selectFolder, getStartStop
from position import plotMultiBeadColor
import matplotlib.pyplot as plt
import pandas as pd
import math

pos = ([], [], [])
val = []

def buildPart(e):
    id='Current(A)'
    avgScale=int(48000/20)

    df = pd.read_csv(e.path + '/aligned_data.csv')

    start, stop = getStartStop(df['Avg_Voltage(V)'], 1)
    df = df[start:stop]
    df = df.reset_index()

    for i in range(math.floor(len(df['time']) / avgScale) - 1):
        pos[0].append((df['Pos_x(mm)'][i*avgScale], df['Pos_x(mm)'][(i+1)*avgScale]))
        pos[1].append((df['Pos_y(mm)'][i*avgScale], df['Pos_y(mm)'][(i+1)*avgScale]))
        pos[2].append((df['Pos_z(mm)'][i*avgScale], df['Pos_z(mm)'][(i+1)*avgScale]))

        val.append(df[id][i*avgScale:(i+1)*avgScale].mean())

    final = math.floor(len(df['time'])/ avgScale)
    pos[0].append((df['Pos_x(mm)'][final*avgScale], df['Pos_x(mm)'].iloc[-1]))
    pos[1].append((df['Pos_y(mm)'][final*avgScale], df['Pos_y(mm)'].iloc[-1]))
    pos[2].append((df['Pos_z(mm)'][final*avgScale], df['Pos_z(mm)'].iloc[-1]))

    val.append(df[id][final*avgScale:].mean())

    return

def main():
    f = selectFolder()

    dataSearch(f, buildPart)

    plotMultiBeadColor(pos, val)
    plt.show()

    return

if __name__ == "__main__": main()