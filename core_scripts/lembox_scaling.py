import pandas as pd
import os
import sys
from pathlib import Path
import numpy as np

from core_scripts.helper_functions import selectFolder

# TODO: add **kwargs to allow for modification of input/output filenames
def scale_lembox(dir, save_cols=['Timestamp', 'Scaled_Voltage(V)', 'Scaled_Current(A)']):

    """ 
    Scales raw lembox_data.csv files to physical voltage and current values. Stores raw data
    in a safe location to avoid future modification/overwrites.

    Args:
        dir (str, pathlib.Path): the 'data_collection_' directory path containing 'lembox_data.csv'
        
        save_cols (list, optional): List of DataFrame columns to store in the output file. Defaults 
        to ['Timestamp', 'Scaled_Voltage(V)', 'Scaled_Current(A)'].
    """

    dir = Path(dir)

    if not os.access(dir / 'raw_data', os.R_OK):
        os.mkdir(dir / 'raw_data')

    input_file = dir / 'raw_data' / 'lembox_data__raw.csv'
    output_file = dir / 'lembox_data.csv'

    if not os.access(dir / 'lembox_data.csv', os.R_OK):
        print(f'Error: {dir / 'lembox_data.csv'} is not a valid lembox data file.\n')
        sys.exit(1)

    # move raw data file to safe location
    os.replace(dir / 'lembox_data.csv', input_file)

    print(f"Reading and updating file: {input_file}")

    # Check raw data file for 'Samples' string. This indicates that the LEMBOX data recording function has injected a stdout 
    # string into the data file (known issue). To resolve, we remove the lines before & after the bad line, + the bad line itself (minimal data loss)

    with open(input_file, 'r') as f:
        rm = []
        r = f.readlines()

        for i, line in enumerate(r[1:]):
            numCommas = line.count(',')
            invalidCommas = numCommas != 6 and numCommas != 8 and numCommas != 11
            if 'Sample' in line or invalidCommas:
                rm.append(i + 1)

        rm.sort(reverse=True)
        for i in rm:
            r.pop(i)

    cols = tuple(r[0][:-1].split(','))
    dat = [tuple(row[:-1].split(',')) for row in r[1:]]


    # added explicit dtype definitions to prevent dtypeWarning message.
    col_types = {
        'Sample': np.int32,
        'PerfTime(s)': np.float64,
        'Timestamp': str,
        'VoltageRaw': str,
        'Voltage(V)': np.float64,
        'CurrentRaw': str,
        'Current(A)': np.float64,
    }

    df = pd.DataFrame(dat, columns=cols)
    df = df.astype(dtype=col_types)

    # scale and update the data in place
    df['Scaled_Voltage(V)'] = df['Voltage(V)'] * 10
    df['Scaled_Current(A)'] = df['Current(A)'] * 100

    df = df[save_cols]

    # Save back to the same file
    df.to_csv(output_file, index=False)
    print(f"Scaling complete. New file created: {output_file}")
        

if __name__ == '__main__':
    dir = selectFolder()
    scale_lembox(dir)