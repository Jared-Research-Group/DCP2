from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import cv2

from data_manipulation import get_FLIR_model

from create_flirvideo import convert_to_8bit, apply_inverted_colormap, add_vertical_color_scale_bar

if __name__ == "__main__":
    temp_file = filedialog.askopenfilename()

    temp_data = np.loadtxt(temp_file, delimiter=',')
    temp_data = cv2.flip(temp_data, 1)
    temp_data = cv2.rotate(temp_data, cv2.ROTATE_90_COUNTERCLOCKWISE)

    model = get_FLIR_model("E:/410SS DATA/modified datasets/L wall 100C Interpass 121125/Mason Print 12_11_25/test/data_collection_20251211_150623/FLIR")

    temp_data = convert_to_8bit(temp_data, np.min(temp_data), np.max(temp_data))

    temp_data = add_vertical_color_scale_bar(temp_data, 464, 348, 7000, 2**16 - 1, model)

    frame_file = filedialog.askopenfilename()

    frame_data = plt.imread(frame_file)

    #raw_folder = filedialog.askdirectory()

    #model = get_FLIR_model(raw_folder)

    #max_temp = model(2**16 - 1)

    fig, ax = plt.subplots(2,1)

    print(temp_data[0][0])
    print()

    print(frame_data[0][0])


    ax[0].imshow(temp_data)
    ax[1].imshow(frame_data)

    plt.show()



