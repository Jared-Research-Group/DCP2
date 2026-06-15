import pandas as pd
import os
import sys
import pathlib
import numpy as np

# Add build directory to path so compiled Cython modules can be found
build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'build', 'lib.win-amd64-cpython-311')
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

from data_manipulation import selectFolder

def scale_lembox(dir, save_cols=['Timestamp', 'Scaled_Voltage(V)', 'Scaled_Current(A)']):

    dir = pathlib.Path(dir)

    if not os.access(dir / 'raw_data', os.R_OK):
        os.mkdir(dir / 'raw_data')

    input_file = dir / 'raw_data' / 'lembox_data.csv'
    output_file = dir / 'lembox.csv'

    if not os.access(input_file, os.R_OK):
        print(f'Error: {input_file} is not a valid lembox data file.\n')
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

    cols = tuple(r[0].split())
    dat = (tuple(row.split()) for row in r[1:])


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

    df = pd.DataFrame(dat, columns=cols, dtype=col_types)

    # scale and update the data in place
    df['Scaled_Voltage(V)'] = df['Voltage(V)'] * 10
    df['Scaled_Current(A)'] = df['Current(A)'] * 100

    df = df[save_cols]

    # Save back to the same file
    df.to_csv(output_file, index=False)
    print(f"Scaling complete. New file created: {output_file}")
        

if __name__ == '__main__':
    dir = selectFolder()
    scaleLembox(dir)