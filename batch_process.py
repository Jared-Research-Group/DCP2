import os
import sys
import subprocess

from core_scripts.helper_functions import selectFolder

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
                elif entry.is_dir():
                    subsearch_discoveries = dataSearch(entry.path, func, printFlag, id, id_atFront)
                
                    if subsearch_discoveries:
                        for e in subsearch_discoveries:
                            found_entities.append(e)

    if printFlag: print(found_entities)
    return found_entities

def batchOperation(e):
    from align_data import alignData

    process_script_dir = os.getcwd()
    subprocess.run([sys.executable, process_script_dir + '\\core_scripts/create_flirvideo.py', e.path + '\\FLIR', e.path + '\\FLIR.mp4', e.path + '\\FLIR_Frames'], check=True)

    return

if __name__ == '__main__':

    from data_preprocessing import process_data_folder

    parent = selectFolder()
    dataSearch(parent, process_data_folder)