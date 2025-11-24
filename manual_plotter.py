import matplotlib.pyplot as plt
import matplotlib        as mpl
import pandas as pd
from tkinter import filedialog
import numpy as np



def drawArcPos(dir, v, i, avgV, avgI, p, t, t_scale):
    
    fig, ax = plt.subplots(2,2, sharex=True, layout='constrained')
    mpl.rcParams['lines.markersize'] = 0.005

    ax[0,0].scatter(p, v)
    ax[0,0].set_title('Voltage')
    ax[0,0].set_ylabel('Voltage (V)')
    #ax[0,0].set_xscale('log')

    ax[1,0].scatter(p, avgV)
    ax[1,0].set_title('Rolling Voltage Average Over ' + t_scale + ' Seconds')
    ax[1,0].set_ylabel('Average Voltage (V)')
    ax[1,0].set_xlabel('y-displacement (mm)')
    #ax[1,0].set_xscale('log')

    ax[0,1].scatter(p, i)
    ax[0,1].set_title('Current')
    ax[0,1].set_ylabel('Current (A)')
    #ax[0,1].set_xscale('log')

    ax[1,1].scatter(p, avgI)
    ax[1,1].set_title('Rolling Current Average Over ' + t_scale + ' Seconds')
    ax[1,1].set_ylabel('Average Current (A)')
    ax[1,1].set_xlabel('y-displacement (mm)')
    #ax[1,1].set_xscale('log')
    plt.locator_params('x', nbins=52)
    
    
    fig.set_size_inches(30,15)
    #plt.savefig(dir + '/visualizations/arcPosY.png')
    
    fig, ax = plt.subplots(layout='constrained')
    ax.scatter(t, p)
    ax.set_ylabel('y-displacement(mm)')
    ax.set_xlabel('Time (s)')
    fig.set_size_inches(15,10)
    
    
    return 

def main():
    
    dir = filedialog.askdirectory()
    df = pd.read_csv(dir + '/aligned_data.csv')
    
    startTime = 0
    stopTime = 0
    
    stage2 = 0
    stage3 = 0
    stage4 = 0
    posWallEdge = 0
    while df['Pos_y(mm)'][startTime] == 0: startTime += 1
    while df['time'][stage2] < 22.5: stage2 += 1
    while df['time'][stage3] < 30.0: stage3 += 1
    while df['time'][stage4] < 32.5: stage4 += 1
    while -1 * df['Pos_y(mm)'][posWallEdge] < (11.563/2): posWallEdge += 1
    while df['Pos_y(mm)'][stopTime] != df['Pos_y(mm)'].iloc[-1]: stopTime += 1
    
    #print('Torch y-displacement at 22.5s: ' + str(-1 * df['Pos_y(mm)'][startTime]))                 // only true when 2nd while loop active
    #print(df['time'][startTime])
    #print(df['Pos_y(mm)'][stopTime])
    #print(df['time'][stopTime])
    
    print(df['time'][startTime])
    print(-1 * df['Pos_y(mm)'][startTime])
    print(df['time'][stage2])
    print(-1 * df['Pos_y(mm)'][stage2])
    print(df['time'][stage3])
    print(-1 * df['Pos_y(mm)'][stage3])
    print(df['time'][stage4])
    print(-1 * df['Pos_y(mm)'][stage4])
    print()
    print(df['time'][posWallEdge])
    
    avgScale = (48000/20000) * 5000
    t_scale = df['time'][int(len(df['time'])/2 + avgScale)] - df['time'][int(len(df['time'])/2)]
    
    drawArcPos(dir, df['Voltage(V)'][startTime:stopTime], df['Current(A)'][startTime:stopTime], df['Avg_Voltage(V)'][startTime:stopTime], df['Avg_Current(A)'][startTime:stopTime], -1*df['Pos_y(mm)'][startTime:stopTime], df['time'][startTime:stopTime], str(t_scale))
    plt.show()
    
    return

if __name__ == '__main__': main()

