import os
import sys
from pathlib import Path
from functools import partial
import shutil

import batch_process
import helper_functions

logger = helper_functions.setup_logger(__name__)

from lembox_scaling    import scale_lembox
from robotdata_parsing import convert_robot_data_to_csv
from audio_conversion  import csv_to_wav
from create_flirvideo  import npy_to_video
from thermocouple      import preprocess_thermocouple
from xiris             import buildVideoCV2
from rebase_time       import temporal_alignment

def process_data_folder(folder_path, forceUpdate=False):
    """Process different types of data files in the given folder using appropriate scripts."""
    folder = Path(folder_path)
    script_dir = Path(__file__).parent
    
    # Dictionary mapping file patterns to their processing scripts
    processing_rules_raw = {
        'microphone_data.csv': csv_to_wav,
        'robot_data.txt': convert_robot_data_to_csv,
        'lembox_data.csv': scale_lembox,
        'FLIR': npy_to_video,
        'thermocouple_data.csv': preprocess_thermocouple,
        'Xiris': buildVideoCV2,
    }

    processing_rules_clean = {
        'microphone_data__raw.csv': csv_to_wav,
        'robot_data__raw.csv': convert_robot_data_to_csv,
        'lembox_data__raw.csv': scale_lembox,
        'thermocouple_data__raw.csv': preprocess_thermocouple,
        'FLIR__raw': npy_to_video, 
        'Xiris__raw': buildVideoCV2
    }
    
    processed = set()

    # Process regular files
    for file_path in folder.glob('*'):
        name = file_path.name
        func = None
        
        # check for 
        if name in processing_rules_raw:
            func = processing_rules_raw[name]
        elif name in processing_rules_clean and forceUpdate:
            func = processing_rules_clean[name]
        
        if func is not None:
            logger.debug(f"Processing {file_path} with {func.__name__}")

            try:
                func(folder)
                    
            except Exception as e:
                logger.error(f"Error running {func.__name__}: {e}")

        processed.add(name)

    # need a better entry point
    if any([n.startswith(('lembox', 'microphone')) for n in processed]):
        temporal_alignment(folder.parents[1] / 'modified_data' / folder.name, ignition_wait_time=.2, flag_plot=True)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        folder_path = sys.argv[1]
    else:
        print("Please select the data collection folder...")
        folder_path = helper_functions.selectFolder()
    
    if not folder_path:
        logger.critical("No folder selected. Exiting...")
        sys.exit(1)
    
    if not os.path.isdir(folder_path):
        logger.critical(f"Error: {folder_path} is not a valid directory")
        sys.exit(1)

    folder_path = Path(folder_path)
    
    # if we are in a data_collection_ directory, preprocess the data in the directory
    if True in [(item.name.endswith('data.csv') or item.name.endswith('data__clean.csv')) for item in folder_path.iterdir()]:

        if folder_path.parents[0].name != 'raw_data':
            for item in [item for item in folder_path.parents[0].iterdir()]:
                shutil.move(item, folder_path.parents[0] / 'raw_data' / item.name)
            folder_path = folder_path.parents[0] / 'raw_data' / folder_path.name

        logger.info(f"Processing folder: {folder_path}")
        process_data_folder(folder_path, forceUpdate=True) # always rerun analysis if we are targeting a single folder

    # if we are not in a data_collection_ directory, recursively search for data collection directories and apply preprocessing to them
    else:
        
        if True in [item.name.startswith('data_collection_') for item in folder_path.iterdir()]:
            logger.info('Creating build directory structure...')
            for item in [item for item in folder_path.iterdir()]:
                shutil.move(item, folder_path / 'raw_data' / item.name)

        folder_path = folder_path / 'raw_data'

        logger.info(f"Processing build: {folder_path.parent}")
        batch_process.dataSearch(folder_path, partial(process_data_folder, forceUpdate=True)) # don't rerun analysis if we target a tree of directories (save time)
