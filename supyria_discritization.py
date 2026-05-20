
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog

def get_slices(csv_filename):
    df = pd.read_csv(csv_filename)
    current = df['Current(A)']
    weld_start_index = current[current > 10].index[0]  #find weld start
    weld_end_index = current[current > 10].index[-1]  #find weld end
    weld_length = df['Pos_x(mm)'][weld_end_index] - df['Pos_x(mm)'][weld_start_index]
    print('weld length: ' + str(weld_length))
    

    slice_end_dist = 0
    i = 0

    slice_df_folder = csv_filename[:-4] + '_slices'
    os.makedirs(slice_df_folder, exist_ok=True)

    max_pos = df['Pos_x(mm)'][weld_end_index]
    print('entering slicing loop...')
    while True:
        slice_start_dist = df['Pos_x(mm)'][weld_start_index] + i * 10

        if slice_start_dist >= max_pos:
            print("Reached end of weld.")
            break

        slice_end_dist = slice_start_dist + 10
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
            f"{os.path.basename(csv_filename)[:-4]}_slice_{i}.csv")

        slice_df.to_csv(output_file, index=False)

        i += 1

    print('Slices saved to ' + slice_df_folder)
    return slice_df_folder




def slice_stats(slice_df_folder):

    if slice_df_folder is None or not os.path.exists(slice_df_folder):
        print("No slices found.")
        return

    slice_files = [f for f in os.listdir(slice_df_folder) if f.endswith('.csv')]
    for slice_file in slice_files:
        slice_df = pd.read_csv(os.path.join(slice_df_folder, slice_file))

        avg_current = slice_df['Current(A)'].mean()
        avg_voltage = slice_df['Voltage(V)'].mean()
        avg_power = (slice_df['Current(A)'] * slice_df['Voltage(V)']).mean()
        avg_velocity = slice_df['Vel_Comb(mm/s)'].mean()
        heat_input = (avg_power)/avg_velocity # J/mm
        stats_df = pd.DataFrame({
            'Slice File': [slice_file],
            'Average Current (A)': [avg_current],
            'Average Voltage (V)': [avg_voltage],
            'Average Power (W)': [avg_power],
            'Average Velocity (mm/s)': [avg_velocity],
            "Heat Input (J/mm)": [heat_input]
        })
        print(f'\nStats for {slice_file}:')
        print(stats_df.to_string(index=False))
        stats_df.to_csv(os.path.join(slice_df_folder, 'slice_stats.csv'), mode='a', index=False, header=not os.path.exists(os.path.join(slice_df_folder, 'slice_stats.csv')))


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


if __name__ == "__main__":
    main()
