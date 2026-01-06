import os
import sys
import subprocess
import tkinter as tk
from tkinter import filedialog

from data_manipulation import selectFolder

def dataSearch(f, func, printFlag=True, id='data_collection_', id_atFront=True):
    found_entities = []

    with os.scandir(f) as it:
        for entry in it:
                if printFlag: print(entry.name)

                if id_atFront: id_found = entry.name.startswith(id)
                else: id_found = entry.name.endswith(id)

                if id_found:
                    if printFlag: print('Found target ' + entry.path)
                    func(entry)
                    found_entities.append(entry.path)
                #elif entry.name.startswith('Layer'):
                    #if count < complete: count, complete = dataSearch(entry.path, func, count, complete)
                    #count += 1
                elif entry.is_dir():
                    subsearch_discoveries = dataSearch(entry.path, func, printFlag, id, id_atFront)
                
                    if subsearch_discoveries:
                        for e in subsearch_discoveries:
                            found_entities.append(e)

    if printFlag: print(found_entities)
    return found_entities

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