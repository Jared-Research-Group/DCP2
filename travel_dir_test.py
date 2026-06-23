import sys
import os
import cv2 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import tkinter as tk
from tkinter import filedialog
from moviepy import VideoFileClip
#Script to slice weld data into slices of desired length and calculate stats for each slice. Stats are saved to a csv file in the same folder as the slices. Stats include average current, voltage, power, velocity, heat input, and slice start/end time.
#Currently only valid for staright beads in x direction, some work could be done to geranlize for beads in x or y diretions. Change unit lenght variable to change slice length. 

def get_slices(csv_filename):
    df = pd.read_csv(csv_filename)
    print(csv_filename)
    current = df['Current(A)']
    weld_start_index = current[current > 10].index[0]  #find weld start
    weld_end_index = current[current > 10].index[-1]  #find weld end
    # find weld direction
    sum_vel_x = df['Vel_x(mm/s)'].sum()
    sum_vel_y = df['Vel_y(mm/s)'].sum()
    if abs(sum_vel_x) > abs(sum_vel_y):
        weld_direction = 'x'
        sum_vel = sum_vel_x
    else:
        weld_direction = 'y'
        sum_vel = sum_vel_y 
    pos_neg = 'positive' if sum_vel > 0 else 'negative'
    
    
    print('weld direction: ' + pos_neg + ' '+ weld_direction)
    
    
    
    #weld_length = df['Pos_x(mm)'][weld_end_index] - df['Pos_x(mm)'][weld_start_index]
    #print('weld length: ' + str(weld_length))
 
    #unit_length = 10  #mm
    #max_pos = df['Pos_x(mm)'][weld_end_index]
    #if weld_length < 0:
    #    print("Weld length is negative.")
    #    unit_length = -unit_length
    #    max_pos = df['Pos_x(mm)'][weld_end_index]
    #    return

    #slice_end_dist = []
    #i = 0
    #df = df.iloc[weld_start_index:weld_end_index+1].reset_index(drop=True)  # Reset index to start from 0

    #slice_df_folder = csv_filename[:-4] + '_slices'
    #os.makedirs(slice_df_folder, exist_ok=True)



def main():
    if len(sys.argv) != 2:
        csv_filename = filedialog.askopenfilename()
    else:
        csv_filename = sys.argv[1]

    if not csv_filename:
        print('No file selected.')
        return

    get_slices(csv_filename)

    


if __name__ == "__main__":
    main()
