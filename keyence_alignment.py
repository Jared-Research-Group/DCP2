import pandas as pd
import matplotlib.pyplot as plt

from data_manipulation import getStartStop, dfToCsv

def readKeyenceData(filename, stop_height):
    df = pd.read_csv(filename)
    
    df = df.dropna(ignore_index=True)
    
    # find the measured center point of stop crater
    stop_index = 0
    while df['Z(mm)'][stop_index] != stop_height:
        stop_index += 1
    
    # find a point 4 inches away from the stop point (assumed starting point)
    start_x = df['X(mm)'][stop_index] + (25.4*4)
    if start_x > df['X(mm)'].iloc[-1]:
        start_index = len(df['X(mm)']) - 1
    else:                                             # if assumed start point is outside frame, assume first point is start point (this will require rescanning plate really)
        start_index = stop_index
        while df['X(mm)'][start_index] < start_x:
            start_index += 1
    
    # recompute x values to be distance from start point
    start_x = df['X(mm)'][start_index]
    for i, x in enumerate(df['X(mm)']):
        df.loc[i, 'X(mm)'] = start_x - x
    
    # reverse df to be ascending from start point
    df = df.iloc[::-1].reset_index(drop=True)
    
    return df

def alignProfileData():
    keyence_file = "E:/mason's stuff/011925 Keyence 4 Jared/bead4.csv"
    bead_dat_file = "F:/410SS DATA/modified datasets/Day_2/data_collection_20251113_155758/aligned_data.csv"
    
    profile = readKeyenceData(keyence_file, 1.525)
    bead    = pd.read_csv(bead_dat_file)
    
    weld_start, weld_stop = getStartStop(bead['Avg_Voltage(V)'], 1)
    weld_start_pos = bead['Pos_x(mm)'][weld_start]
    
    for i, x in enumerate(profile['X(mm)']):
        profile.loc[i, 'X(mm)'] = x + weld_start_pos
        
    bead['profile_x(mm)'] = pd.NA
    bead['profile_z(mm)'] = pd.NA
    
    print(profile)
    
    current_index = 0 
    for i in range(weld_start, weld_stop):
        #print(current_index)
        while profile['X(mm)'][current_index + 1] < bead['Pos_x(mm)'][i] and current_index < len(profile['X(mm)']) - 2:
            current_index += 1
        
        bead.loc[i, 'profile_x(mm)'] = profile['X(mm)'][current_index]
        bead.loc[i, 'profile_z(mm)'] = profile['Z(mm)'][current_index]
        
    bead = bead.dropna(subset=['profile_x(mm)', 'profile_z(mm)'])
    
    dfToCsv(bead, "F:/410SS DATA/profile_datasets/bead4.csv")

def drawVis():
    bead = pd.read_csv("F:/410SS DATA/profile_datasets/bead3.csv")
    
    fig1, ax1 = plt.subplots(2,1, layout='constrained')
    ax1[0].scatter(bead['profile_z(mm)'], bead['Avg_Voltage(V)'], s=0.000005)
    ax1[0].set_ylabel('Average Voltage (V)')
    
    ax1[1].scatter(bead['profile_z(mm)'], bead['Avg_Current(A)'], s=0.000005)
    ax1[1].set_ylabel('Average Current (A)')
    ax1[1].set_xlabel('Bead Profile Height (mm)')
    fig1.suptitle('Arc Data vs Bead Height')
    
    fig2, ax2 = plt.subplots(layout='constrained')
    ax2.scatter(bead['profile_x(mm)'], bead['profile_z(mm)'], s=0.005)
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel('X Position Along Bead (mm)')
    ax2.set_ylabel('Bead Profile Height (mm)')
    ax2.set_title('Bead Profile')
    plt.show()

if __name__ == '__main__':

    drawVis()