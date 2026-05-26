import matplotlib.pyplot as plt
import pandas as pd

from data_manipulation import selectFolder, getStartStop

def val_plot(dir):
    df = pd.read_csv(dir + '/aligned_data.csv')

    startTime, stopTime = getStartStop(df['Avg_Voltage(V)'], 1)

    startTime = max(startTime - 96000, 0)
    stopTime = min(stopTime + 96000, len(df['Avg_Voltage(V)']))

    print(startTime)
    print(stopTime)

    plt.scatter(df['time'][startTime:stopTime], df['Current(A)'][startTime:stopTime], s=0.005)
    #plt.scatter(df['time'][startTime + 3*48000:startTime + 4*48000], df['Current(A)'][startTime + 3*48000:startTime + 4*48000], s=0.01)
    plt.show()

if __name__ == '__main__':
    #dir = selectFolder()
    dir = "C:/Users/wwerner4/Documents/data/Han Test (Last in Han2)/data_collection_20260327_142705"

    val_plot(dir)