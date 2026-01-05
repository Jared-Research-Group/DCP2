import os
import sys
import subprocess
import tkinter as tk
from tkinter import filedialog

from data_manipulation import selectFolder

def dataSearch(f, func, printFlag=True, id='data_collection_'):

    with os.scandir(f) as it:
        for entry in it:
                if printFlag==True: print(entry.name)
                if entry.name.startswith(id):
                    if printFlag==True: print('Found target ' + entry.path)
                    func(entry)
                #elif entry.name.startswith('Layer'):
                    #if count < complete: count, complete = dataSearch(entry.path, func, count, complete)
                    #count += 1
                elif entry.is_dir():
                    dataSearch(entry.path, func, printFlag, id)

    return

def batchOperation(e):

    process_script_dir = os.getcwd()
    #subprocess.run([sys.executable, process_script_dir + '\\create_flirvideo.py', e.path + '\\FLIR', e.path + '\\FLIR.mp4', e.path + '\\FLIR_Frames'], check=True)
    subprocess.run([sys.executable, process_script_dir + '\\Data_Processing.py', e.path], check=True)
    if os.access(e.path + '\\robot_data.csv', os.R_OK): subprocess.run([sys.executable, process_script_dir + '\\synchronized_data.py', e.path], check=True)

    return

def main():

    parent = selectFolder()

    dataSearch(parent, batchOperation)

if __name__ == '__main__':
    main()