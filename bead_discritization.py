
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
    weld_length = df['Pos_x(mm)'][weld_end_index] - df['Pos_x(mm)'][weld_start_index]
    print('weld length: ' + str(weld_length))
 
    unit_length = 10  #mm
    max_pos = df['Pos_x(mm)'][weld_end_index]
    if weld_length < 0:
        print("Weld length is negative.")
        unit_length = -unit_length
        max_pos = df['Pos_x(mm)'][weld_end_index]
        return

    slice_end_dist = []
    i = 0
    df = df.iloc[weld_start_index:weld_end_index+1].reset_index(drop=True)  # Reset index to start from 0

    slice_df_folder = csv_filename[:-4] + '_slices'
    os.makedirs(slice_df_folder, exist_ok=True)

   
    print('entering slicing loop...')
    while True:
        slice_start_dist = df['Pos_x(mm)'][0] + i * unit_length

        if slice_start_dist >= max_pos:
            print("Reached end of weld.")
            break

        slice_end_dist = slice_start_dist + unit_length
        if slice_end_dist > max_pos:
            slice_end_dist = max_pos

        start_candidates = df[df['Pos_x(mm)'] >= slice_start_dist]
        end_candidates = df[df['Pos_x(mm)'] >= slice_end_dist]


        if start_candidates.empty or end_candidates.empty:
            print("No more valid data for slicing.")
            break

        slice_start = start_candidates.index[0]
        slice_end = end_candidates.index[0]

        print(f"Slice {i}: {slice_start_dist} → {slice_end_dist}")

        slice_df = df.iloc[slice_start:slice_end]

        output_file = os.path.join(
            slice_df_folder,
            f"{os.path.basename(csv_filename)[:-4]}_slice_{i:03d}.csv")

        slice_df.to_csv(output_file, index=False)

        i += 1

    print('Slices saved to ' + slice_df_folder)
    return slice_df_folder


def slice_stats(slice_df_folder):
    print(slice_df_folder)
    if slice_df_folder is None or not os.path.exists(slice_df_folder):
        print("No slices found.")
        return

    slice_files = sorted(
        [f for f in os.listdir(slice_df_folder) if f.endswith('.csv')],
        key=lambda f: int(re.search(r'_(\d+)\.csv$', f).group(1))
    )
    for slice_file in slice_files:
        slice_df = pd.read_csv(os.path.join(slice_df_folder, slice_file))

        avg_current = slice_df['Current(A)'].mean()
        avg_voltage = slice_df['Voltage(V)'].mean()
        avg_power = (slice_df['Current(A)'] * slice_df['Voltage(V)']).mean()
        avg_velocity = slice_df['Vel_Comb(mm/s)'].mean()
        heat_input = (avg_power)/avg_velocity # J/mm
        slice_start_time = slice_df['time'].iloc[0]
        slice_end_time = slice_df['time'].iloc[-1]
        
        stats_df = pd.DataFrame({
            'Slice File': [slice_file],
            'Average Current (A)': [avg_current],
            'Average Voltage (V)': [avg_voltage],
            'Average Power (W)': [avg_power],
            'Average Velocity (mm/s)': [avg_velocity],
            "Heat Input (J/mm)": [heat_input],
            'Slice Start Time (s)': [slice_start_time],
            'Slice End Time (s)': [slice_end_time]
        })
        print(f'\nStats for {slice_file}:')
        print(stats_df.to_string(index=False))
        stats_df.to_csv(os.path.join(slice_df_folder, 'slice_stats.csv'), mode='a', index=False, header=not os.path.exists(os.path.join(slice_df_folder, 'slice_stats.csv')))

def FLIR_Video_Slices(FLIR_filename,slice_df_folder):
    flir_df = pd.read_csv(slice_df_folder + '/slice_stats.csv')
    start_times = flir_df['Slice Start Time (s)'] * 3 # Multiplied by three to correct for the FLIR video frame rate being 10fps instead of 30. Need to update FLIR video script. 
    end_times = flir_df['Slice End Time (s)'] * 3 # Multiplied by three to correct for the FLIR video frame rate being 10fps instead of 30. Need to update FLIR video script. 
    print(start_times)
    print(end_times)
    flir_clip = VideoFileClip(FLIR_filename)
    flir_slice_folder = FLIR_filename[:-4] + '_slices'
    os.makedirs(flir_slice_folder, exist_ok=True)
    for i in range(len(start_times)):
        start = start_times[i]
        end = end_times[i]
        sliced_clip = flir_clip.subclipped(start, end)
        sliced_clip.write_videofile(f"{flir_slice_folder}/slice_{i}.mp4")
        print(f"Saved slice {i} as {flir_slice_folder}/slice_{i}.mp4")



def Xiris_Video_Slices(Xiris_filename,slice_df_folder):
    # Determin start time of Xiris Video
    xiris_clip = VideoFileClip(Xiris_filename)
    
    

    xiris_df = pd.read_csv(slice_df_folder + '/slice_stats.csv')
    start_times = xiris_df['Slice Start Time (s)'] 
    end_times = xiris_df['Slice End Time (s)'] 
    print(start_times)
    print(end_times)
    xiris_clip = VideoFileClip(Xiris_filename)
    xiris_slice_folder = Xiris_filename[:-4] + '_slices'
    os.makedirs(xiris_slice_folder, exist_ok=True)
    for i in range(len(start_times)):
        start = start_times[i]
        end = end_times[i]
        sliced_clip = xiris_clip.subclipped(start, end)
        sliced_clip.write_videofile(f"{xiris_slice_folder}/slice_{i}.mp4")
        print(f"Saved slice {i} as {xiris_slice_folder}/slice_{i}.mp4")
    


def main():
    if len(sys.argv) != 2:
        csv_filename = filedialog.askopenfilename()
    else:
        csv_filename = sys.argv[1]

    if not csv_filename:
        print('No file selected.')
        return

    get_slices(csv_filename)

    slice_stats(csv_filename[:-4] + '_slices')

    FLIR_Video_Slices(csv_filename[:-16]+"FLIR.mp4", csv_filename[:-4] + '_slices')

    #Xiris_Video_Slices(csv_filename[:-16]+"Xiris.avi")
    


if __name__ == "__main__":
    main()
