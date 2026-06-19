import os
import numpy as np
import cv2
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

from audio_conversion import csv_to_wav
from create_flirvideo import (
    convert_to_8bit,
    apply_inverted_colormap,
    add_vertical_color_scale_bar,
    add_timestamp,
)

def npy_to_video(flir_dir, video_path, frames_path, log_func, fps=10, size=(640, 480)):
    npy_files = [f for f in os.listdir(flir_dir) if f.endswith('.npy')]
    data_list = []

    for file in npy_files:
        try:
            full_path = os.path.join(flir_dir, file)
            data = np.load(full_path, allow_pickle=True).item()
            timestamp = data.get('timestamp')
            if not timestamp:
                log_func(f"Timestamp not found in {file}. Skipping.")
                continue
            data_list.append((file, data['frame'], timestamp))
        except Exception as e:
            log_func(f"Error processing {file}: {e}")

    if not data_list:
        log_func("No valid FLIR npy files found.")
        return

    try:
        data_list.sort(key=lambda x: datetime.strptime(x[2], '%Y-%m-%d %H:%M:%S.%f'))
    except Exception as e:
        log_func(f"Error sorting files by timestamp: {e}")
        return

    global_min = min(frame.min() for (_, frame, _) in data_list)
    global_max = max(frame.max() for (_, frame, _) in data_list)
    os.makedirs(frames_path, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, size)

    for fname, frame, timestamp in data_list:
        try:
            img = convert_to_8bit(frame, global_min, global_max)
            img = apply_inverted_colormap(img)
            img = cv2.resize(img, size)
            img = add_vertical_color_scale_bar(img, *size, global_min, global_max)
            img = add_timestamp(img, timestamp, *size)
            out.write(img)
            
            frame_file = os.path.join(frames_path, f"{fname}.png")
            cv2.imwrite(frame_file, img)
            log_func(f"Saved FLIR frame: {frame_file}")
        except Exception as e:
            log_func(f"Error processing {fname}: {e}")
    out.release()
    log_func(f"FLIR video saved: {video_path}")

def process_folder(folder, log_func):
    log_func(f"\n=== Processing folder: {folder} ===")
    
    # Process microphone data if present.
    mic_csv = os.path.join(folder, "microphone_data.csv")
    if os.path.exists(mic_csv):
        wav_path = os.path.join(folder, "microphone_data.wav")
        try:
            csv_to_wav(mic_csv, wav_filename=wav_path)
            log_func(f"Microphone data processed: {mic_csv} → {wav_path}")
        except Exception as e:
            log_func(f"Error processing microphone data: {e}")
    else:
        log_func("No microphone_data.csv found in this folder.")
    
    # Process FLIR data if present.
    flir_folder = os.path.join(folder, "FLIR")
    if os.path.exists(flir_folder) and os.path.isdir(flir_folder):
        video_path = os.path.join(folder, "FLIR.mp4")
        frames_path = os.path.join(folder, "FLIR_Frames")
        try:
            npy_to_video(flir_folder, video_path, frames_path, log_func)
        except Exception as e:
            log_func(f"Error processing FLIR data: {e}")
    else:
        log_func("No FLIR folder found in this folder.")

def create_gui():
    root = tk.Tk()
    root.title("Main Folder Data Processor")
    
    # Main folder selection frame.
    main_folder_frame = tk.Frame(root)
    main_folder_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

    selected_main_folder = tk.StringVar()

    def select_main_folder():
        folder = filedialog.askdirectory(title="Select Main Folder Containing Data Subfolders")
        if folder:
            selected_main_folder.set(folder)
            main_folder_label.config(text=f"Main Folder: {folder}")

    tk.Button(main_folder_frame, text="Select Main Folder", command=select_main_folder).pack(side=tk.LEFT, padx=5)
    main_folder_label = tk.Label(main_folder_frame, text="No folder selected")
    main_folder_label.pack(side=tk.LEFT, padx=5)

    # Process button.
    tk.Button(root, text="Process All Subfolders", command=lambda: process_all_subfolders()).pack(pady=5)

    # Log text box with scrollbar.
    log_frame = tk.Frame(root)
    log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    log_text = tk.Text(log_frame, wrap=tk.WORD, height=20)
    log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar = tk.Scrollbar(log_frame, command=log_text.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    log_text.config(yscrollcommand=scrollbar.set)

    # Log function: prints to both text widget and stdout.
    def log(message):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
        full_message = timestamp + message
        log_text.insert(tk.END, full_message + "\n")
        log_text.see(tk.END)
        print(full_message)
        root.update_idletasks()

    def process_all_subfolders():
        main_folder = selected_main_folder.get()
        if not main_folder:
            messagebox.showwarning("No Folder Selected", "Please select a main folder first.")
            return
        
        subfolders = [os.path.join(main_folder, d) for d in os.listdir(main_folder)
                      if os.path.isdir(os.path.join(main_folder, d))]
        if not subfolders:
            log("No subfolders found in the main folder.")
            return
        
        for folder in subfolders:
            process_folder(folder, log)
            log(f"=== Finished processing folder: {folder} ===\n")

    root.mainloop()

if __name__ == "__main__":
    create_gui()