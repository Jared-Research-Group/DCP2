import os
import sys
import subprocess
import tkinter as tk
from tkinter import filedialog

from data_manipulation import selectFolder

def dataSearch(f, func, count, complete):

    with os.scandir(f) as it:
        for entry in it:
            if entry.is_dir():
                print(entry.name)
                if entry.name.startswith('data_collection_'):
                    print('Found data folder ' + entry.path)
                    func(entry)
                #elif entry.name.startswith('Layer'):
                    #if count < complete: count, complete = dataSearch(entry.path, func, count, complete)
                    #count += 1
                else:
                    count, complete = dataSearch(entry.path, func, count, complete)

    return count, complete

def batchOperation(e):

    process_script_dir = os.getcwd()
    #subprocess.run([sys.executable, process_script_dir + '\\create_flirvideo.py', e.path + '\\FLIR', e.path + '\\FLIR.mp4', e.path + '\\FLIR_Frames'], check=True) #do day 6
    subprocess.run([sys.executable, process_script_dir + '\\Data_Processing.py', e.path], check=True)
    if os.access(e.path + '\\robot_data.csv', os.R_OK): subprocess.run([sys.executable, process_script_dir + '\\synchronized_data.py', e.path], check=True)

    return

def main():

    parent = selectFolder()

    dataSearch(parent, batchOperation, 0, 1)

if __name__ == '__main__':
    main()