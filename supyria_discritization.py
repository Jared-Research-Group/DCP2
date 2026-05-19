
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog

def get_slices(csv_filename):
    df = pd.read_csv(csv_filename)
    current = df['Current(A)']
    #find weld start
    weld_start_index = current[current > 10].index[0]
    print('Weld start index: ' + str(weld_start_index))
    print('weld start position: ' + str(df['Pos_x(mm)'][weld_start_index]))
    #find weld end
    weld_end_index = current[current > 10].index[-1]
    print('Weld end index: ' + str(weld_end_index))
    print('weld end position: ' + str(df['Pos_x(mm)'][weld_end_index]))
    weld_length = df['Pos_x(mm)'][weld_end_index] - df['Pos_x(mm)'][weld_start_index]
    print('weld length: ' + str(weld_length))
    
    slice_end_dist = 0 
    i = 0
    while slice_end_dist < df['Pos_x(mm)'][weld_end_index]:
        slice_start_dist = df['Pos_x(mm)'][weld_start_index] +i*10
        slice_end_dist = slice_start_dist +10
        if slice_end_dist > df['Pos_x(mm)'][weld_end_index]:
            slice_end_dist = df['Pos_x(mm)'][weld_end_index]
        slice_start = df[df['Pos_x(mm)'] >= slice_start_dist].index[0]
        slice_end = df[df['Pos_x(mm)'] >= slice_end_dist].index[0]
        print('Slice ' + str(i) + ': ' + str(slice_start_dist) + ' to ' + str(slice_end_dist))
        i += 1
        slice_df = df[slice_start:slice_end]
        slice_df.to_csv(csv_filename[:-4] + '_slice_' + str(i) + '.csv', index=False)

def main():
    if len(sys.argv) != 2:
        csv_filename = filedialog.askopenfilename()
    else:
        csv_filename = sys.argv[1]
    get_slices(csv_filename)
    



if __name__ == "__main__":
    main()
