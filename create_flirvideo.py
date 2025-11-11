import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from datetime import datetime

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

def add_vertical_color_scale_bar(image, width, height, global_min, global_max):
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
    min_temperature = global_min
    max_temperature = global_max

    #Lower Range
    #min_temperature = -0.000000045*global_min*global_min + 0.0095*global_min-85
    #max_temperature = -0.000000045*global_max*global_max + 0.0095*global_max-85
    
    #Upper Range
    #min_temperature = 0.0468*global_min-267.72
    #max_temperature = 0.0468*global_max-267.72
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

    for npy_file in npy_files:
        npy_file_path = os.path.join(input_folder, npy_file)
        try:
            data = np.load(npy_file_path, allow_pickle=True)
            image = data.item()['frame']
            global_min = min(global_min, image.min())
            global_max = max(global_max, image.max())
        except Exception as e:
            print(f"Skipping {npy_file}: {e}")

    return global_min, global_max

def npy_to_video(input_folder, output_file, output_frames_folder, fps=10, width=640, height=480):
    global_min, global_max = find_global_min_max(input_folder)
    
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

    # Process frames
    for i, (npy_file, timestamp) in enumerate(npy_files_with_timestamps):
        npy_file_path = os.path.join(input_folder, npy_file)
        try:
            data = np.load(npy_file_path, allow_pickle=True)
            image = data.item()['frame']
            
            image_8bit = convert_to_8bit(image, global_min, global_max)
            colored_image = apply_inverted_colormap(image_8bit)
            resized_image = cv2.resize(colored_image, (width, height))
            
            image_with_scale_bar = add_vertical_color_scale_bar(resized_image, width, height, global_min, global_max)
            final_image = add_timestamp(image_with_scale_bar, timestamp, width, height)
            
            out.write(final_image)
            frame_filename = os.path.join(output_frames_folder, f"{os.path.splitext(npy_file)[0]}.png")
            cv2.imwrite(frame_filename, final_image)

            if (i + 1) % 100 == 0:
                print(f"[{i+1}/{len(npy_files_with_timestamps)}] Saved frame: {frame_filename}")
            
        except Exception as e:
            print(f"Error processing {npy_file}: {e}")

    out.release()
    print(f"Video saved to {output_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python create_flirvideo.py <input_folder> <output_video> <output_frames>")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_video = sys.argv[2]
    output_frames = sys.argv[3]
    
    npy_to_video(input_folder, output_video, output_frames)