import os
import sys
from pathlib import Path
from tkinter import filedialog
import tkinter as tk

from helper_functions  import selectFolder
from lembox_scaling    import scale_lembox
from robotdata_parsing import convert_robot_data_to_csv
from audio_conversion  import csv_to_wav
from create_flirvideo  import npy_to_video
from thermocouple      import preprocess_thermocouple

# TODO: check for existance of data in raw_data/, if it exists, then do not overwrite!
def process_data_folder(folder_path):
    """Process different types of data files in the given folder using appropriate scripts."""
    folder = Path(folder_path)
    script_dir = Path(__file__).parent
    
    # Dictionary mapping file patterns to their processing scripts
    processing_rules = {
        'microphone_data.csv': csv_to_wav,
        'robot_data.txt': convert_robot_data_to_csv,
        'lembox_data.csv': scale_lembox,
        'FLIR': npy_to_video,
        'thermocouple_data.csv': preprocess_thermocouple
    }

    if not os.access(folder / 'raw_data', os.R_OK):
        os.mkdir(folder / 'raw_data')
    
    # Process regular files
    for file_path in folder.glob('*'):
        name = file_path.name
        
        if name in processing_rules:
            func = processing_rules[name]
            
            print(f"\nProcessing {file_path} with {func.__name__}")

            try:
                func(folder)
                    
            except Exception as e:
                print(f"Error running {func.__name__}: {e}")

if __name__ == "__main__":
    if len(sys.argv) == 2:
        folder_path = sys.argv[1]
    else:
        print("Please select the data collection folder...")
        folder_path = selectFolder()
    
    if not folder_path:
        print("No folder selected. Exiting...")
        sys.exit(1)
    
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory")
        sys.exit(1)
    
    print(f"\nProcessing folder: {folder_path}")
    process_data_folder(folder_path)