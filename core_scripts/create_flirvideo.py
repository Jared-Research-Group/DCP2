import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from datetime import datetime
import math
import pysr
import json
import sys
import tkinter as tk
import sympy

# Add build directory to path so compiled Cython modules can be found
build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'build', 'lib.win-amd64-cpython-310')
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

from tkinter import simpledialog
from data_manipulation import selectFolder, printProgressBar

high_fit = pysr.PySRRegressor().from_file(run_directory=os.getcwd() + '/FLIR_fits/High', model_selection='best')
low_fit = pysr.PySRRegressor().from_file(run_directory=os.getcwd() + '/FLIR_fits/Low', model_selection='best')

def convert_to_8bit(image, global_min, global_max):
    image_normalized = (image - global_min) / (global_max - global_min)
    image_normalized = np.clip(image_normalized, 0, 1)
    image_8bit = (image_normalized * 255).astype(np.uint8)
    return image_8bit

def apply_inverted_colormap(image_8bit):
    colormap = plt.get_cmap('jet')
    inverted_colormap = colormap.reversed()
    colored_image = inverted_colormap(image_8bit / 255.0)
    return (colored_image[:, :, :3] * 255).astype(np.uint8)

def add_vertical_color_scale_bar(image, width, height, global_min, global_max, model):
    bar_height = int(0.5 * height)
    bar_thickness = 20
    bar_x_start = width - 30
    bar_y_start = (height - bar_height) // 2

    gradient = np.linspace(0, 1, bar_height)
    gradient_colormap = plt.get_cmap('jet')(gradient)
    gradient_colormap = (gradient_colormap[:, :3] * 255).astype(np.uint8)

    for i in range(bar_height):
        image[bar_y_start + i, bar_x_start:bar_x_start + bar_thickness] = gradient_colormap[i]

###################################################################################################
# This section is where you set the minimum and maximum values of the scale bar.    
    # Test Range (using raw values directly)
    #min_temperature = global_min
    #max_temperature = global_max

    
    min_temperature = model(global_min) - 273.15
    max_temperature = model(global_max) - 273.15

    #min_temperature  = ((((2 * global_min) + math.log(global_min + -1470.9462)) + (global_min * -1.9999775)) * 65.06531) + -329.85684 - 273.15
    #max_temperature  = ((((2 * global_max) + math.log(global_max + -1470.9462)) + (global_max * -1.9999775)) * 65.06531) + -329.85684 - 273.15

###################################################################################################
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)
    thickness = 1

    min_label_pos = (bar_x_start - 45, bar_y_start + bar_height)
    max_label_pos = (bar_x_start - 45, bar_y_start)

    image = cv2.putText(image, f'{min_temperature:.2f}C', min_label_pos, font, font_scale, font_color, thickness)
    image = cv2.putText(image, f'{max_temperature:.2f}C', max_label_pos, font, font_scale, font_color, thickness)

    return image

def add_timestamp(image, timestamp, width, height):
    formatted = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f') \
                      .strftime('%H:%M:%S.%f')[:-3]
    text = f" {formatted}" # Add build label here if desired
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color      = (255, 255, 255)
    thickness  = 2
    size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    loc  = (width - size[0] - 10, height - 10)
    return cv2.putText(image, text, loc, font, font_scale, color, thickness, lineType=cv2.LINE_AA)

def find_global_min_max(input_folder):
    npy_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.npy')])
    global_min = float('inf')
    global_max = float('-inf')

    print('Finding global min/max...')

    for i, npy_file in enumerate(npy_files):
        npy_file_path = os.path.join(input_folder, npy_file)
        try:
            data = np.load(npy_file_path, allow_pickle=True)
            image = data.item()['frame']
            global_min = min(global_min, image.min())
            global_max = max(global_max, image.max())

            printProgressBar(i, len(npy_files))

        except Exception as e:
            print(f"Skipping {npy_file}: {e}")

    return global_min, global_max

def intensity_to_temperature(fr, model):

    temps = np.zeros(fr.shape)

    for i, row in enumerate(fr):
        temps[i] = model(row) - 273.15

    return temps


def npy_to_video(input_folder, output_file, output_frames_folder, forceUpdate=False, fps=10, width=640, height=480):

    print(1)
    if not os.access(output_file, os.R_OK) or forceUpdate:
        global_min, global_max = find_global_min_max(input_folder)

        cal_curve = get_model(input_folder)
        
        # Extract timestamps and sort files
        npy_files_with_timestamps = []
        for npy_file in os.listdir(input_folder):
            if npy_file.endswith('.npy'):
                npy_file_path = os.path.join(input_folder, npy_file)
                try:
                    data = np.load(npy_file_path, allow_pickle=True)
                    timestamp = data.item().get('timestamp', None)
                    if timestamp:
                        npy_files_with_timestamps.append((npy_file, timestamp))
                except Exception as e:
                    print(f"Skipping {npy_file}: {e}")
        
        # Sort by timestamp
        npy_files_with_timestamps.sort(key=lambda x: datetime.strptime(x[1], '%Y-%m-%d %H:%M:%S.%f'))

        # Set up video writer and output folder
        os.makedirs(output_frames_folder, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        print('\nGenerating FLIR Video...')

        # Process frames
        for i, (npy_file, timestamp) in enumerate(npy_files_with_timestamps):
            npy_file_path = os.path.join(input_folder, npy_file)
            try:
                data = np.load(npy_file_path, allow_pickle=True)
                image = data.item()['frame']

                temp_image = intensity_to_temperature(image, cal_curve)
                
                image_8bit = convert_to_8bit(temp_image, cal_curve(global_min) - 273.15, cal_curve(global_max) - 273.15)
                colored_image = apply_inverted_colormap(image_8bit)
                resized_image = cv2.resize(colored_image, (width, height))
                
                # need to compute ends of scale bar correctly. what are the max and min values of a frame/video?
                image_with_scale_bar = add_vertical_color_scale_bar(resized_image, width, height, global_min, global_max, cal_curve)
                final_image = add_timestamp(image_with_scale_bar, timestamp, width, height)
                
                out.write(final_image)
                frame_filename = os.path.join(output_frames_folder, f"{os.path.splitext(npy_file)[0]}.png")
                cv2.imwrite(frame_filename, final_image)

                printProgressBar(i, len(npy_files_with_timestamps))
                
            except Exception as e:
                print(f"\nError processing {npy_file}: {e}")

        out.release()
        print(f"\nVideo saved to {output_file}")
    else:
        print("FLIR video is already created. (Use forceUpdate bool to create anyway)")
    return

class CaseSelectionDialog(simpledialog.Dialog):
    def body(self, master):
        tk.Label(master, text="What temperature range is FLIR data in?").grid(row=0, column=0, sticky="w")

    def buttonbox(self):
        box = tk.Frame(self)

        tk.Button(box, text='High', width=10, command=self.high).pack(side="left", padx=5, pady=5)
        tk.Button(box, text='Low', width=10, command=self.low).pack(side="left", padx=5, pady=5)
        tk.Button(box, text='Cancel', width=10, command=self.cancel).pack(side="left", padx=5, pady=5)

        self.bind("<Return>", self.cancel)
        self.bind("<Escape>", self.cancel)
        box.pack()

    def high(self, event=None):
        self.result = 1
        self.cancel()

    def low(self, event=None):
        self.result = 0
        self.cancel()

def ask_case(parent):
    dialog = CaseSelectionDialog(parent, title='Case Selection')
    return dialog.result
        

def get_model(d_in):

    with open(d_in + '/FLIR_Variables.json', 'r') as json_file:
        params = json.load(json_file)

    if 'case' in params:
        case = int(params['case'])
    else:
        root = tk.Tk()
        root.withdraw()

        case = ask_case(root)                   # dialog needs to close after button press. json assignment doesnt work, and need to rewrite file.
        params["case"] = str(case)

        with open(d_in + '/FLIR_Variables.json', 'w') as json_file:
            json.dump(params, json_file, indent=2)

    x = sympy.symbols('FLIR_Intensity')

    if case == 1:
        #print(high_fit.sympy())
        return sympy.lambdify(x, high_fit.sympy(), modules='numpy')
    else:
        #print(low_fit.sympy())
        return sympy.lambdify(x, low_fit.sympy(), modules='numpy')


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        dir = selectFolder()

        input_folder = dir + '/FLIR'
        output_video = dir + '/FLIR.mp4'
        output_frames = dir + '/FLIR_Frames'
    
    else:

        # Usage: python create_flirvideo.py <input_folder> <output_video> <output_frames>
        input_folder = sys.argv[1]
        output_video = sys.argv[2]
        output_frames = sys.argv[3]
    
    npy_to_video(input_folder, output_video, output_frames, True)