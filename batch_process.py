import os
import sys
import subprocess
from pathlib import Path
from tqdm import tqdm

import helper_functions

logger = helper_functions.setup_logger(__name__)

# recursively search for data directories (i.e., in a parent build directory) and apply a function to the directories
def dataSearch(f, func, progressBar=True, id='data_collection_', id_atFront=True):
    found_entities = []

    f = Path(f)

    # iterate over directory
    for entry in tqdm(list(f.iterdir()), disable= not progressBar):
            logger.debug(entry.name)

            if id_atFront: id_found = entry.name.startswith(id)
            else: id_found = entry.name.endswith(id)

            if id_found:

                logger.info('Found target ' + str(entry))
                func(entry)
                found_entities.append(str(entry))

            elif entry.is_dir():
                subsearch_discoveries = dataSearch(entry, func, progressBar=False, id=id, id_atFront=id_atFront)
            
                if subsearch_discoveries:
                    for e in subsearch_discoveries:
                        found_entities.append(e)

    logger.debug(found_entities)
    return found_entities

def batchOperation(e):
    #from align_data import alignData

    process_script_dir = os.getcwd()
    subprocess.run([sys.executable, process_script_dir + '\\core_scripts/create_flirvideo.py', e.path + '\\FLIR', e.path + '\\FLIR.mp4', e.path + '\\FLIR_Frames'], check=True)

    return

if __name__ == '__main__':

    parent = helper_functions.selectFolder()
    dataSearch(parent, batchOperation)