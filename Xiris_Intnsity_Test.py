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

def get_intensity(Xiris_filename):
    # Load the Xiris video
    cap = cv2.VideoCapture(Xiris_filename)
    intensities = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Calculate the average intensity of the frame
        avg_intensity = np.mean(frame)
        intensities.append(avg_intensity)
    cap.release()
    print(intensities)
    return intensities

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

    get_intensity(csv_filename[:-16]+"Xiris.avi")
    #Xiris_Video_Slices(csv_filename[:-16]+"Xiris.avi")
    


if __name__ == "__main__":
    main()