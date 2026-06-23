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


def Xiris_Video_Slices(Xiris_filename,slice_df_folder):
    print(slice_df_folder)
    # Determin start time of Xiris Video
    cap = cv2.VideoCapture(Xiris_filename)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Calculate the average intensity of the frame
    
        avg_intensity = np.mean(frame)
        if avg_intensity > 15:
            weld_start = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            print(weld_start)
            break

    xiris_df = pd.read_csv(slice_df_folder + '/slice_stats.csv')
    xiris_correction = xiris_df['Slice Start Time (s)'][0] - weld_start
    print(xiris_correction)
    start_times = xiris_df['Slice Start Time (s)'] - xiris_correction
    end_times = xiris_df['Slice End Time (s)'] - xiris_correction
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

    
    Xiris_Video_Slices(csv_filename[:-16]+"Xiris.avi",csv_filename[:-4] + "_slices")
    


if __name__ == "__main__":
    main()