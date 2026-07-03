import pandas as pd
from pathlib import Path
from datetime import timedelta
import os

import batch_process
import helper_functions
import create_flirvideo

def preprocess_aspe_data(dir):
    
    def process_thermocouple(data_dir):
        
        t_file = data_dir / 'thermocouple_data.csv'
        if os.access(t_file, os.R_OK):
        
            data = pd.read_csv(t_file, encoding='cp1252')
            data.to_csv(t_file, index=False)

    batch_process.dataSearch(dir, process_thermocouple, id='deg', id_atFront=False)
    
def preprocess_flir_videos(dir):
    batch_process.dataSearch(dir, create_flirvideo.npy_to_video)
    
def preprocess_pyrometer_data(dir):
    
    def addHour(dir):
        
        dir = Path(dir)
        
        df = pd.read_csv(dir / 'pyrometer.csv', parse_dates=['timestamp'])
        df['timestamp'] = df['timestamp'] + timedelta(hours=1)
        
        df.to_csv(dir / 'pyrometer.csv', index=False)
        
        
    batch_process.dataSearch(dir, addHour, id='deg', id_atFront=False)

if __name__ == '__main__':
    
    dir = helper_functions.selectFolder()
    dir = Path(dir)
    
    preprocess_aspe_data(dir)
    #preprocess_flir_videos(dir / 'flir')
    preprocess_pyrometer_data(dir)