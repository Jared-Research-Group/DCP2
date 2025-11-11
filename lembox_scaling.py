import pandas as pd
import os
import sys
import numpy as np
from data_manipulation import csvHasColumn, dfToCsv

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) > 1:
    # Make paths relative to script directory if it's not absolute
    input_file = os.path.join(script_dir, sys.argv[1]) if not os.path.isabs(sys.argv[1]) else sys.argv[1]
else:
    input_file = os.path.join(script_dir, 'lembox_data.csv')

if not os.access(input_file, os.R_OK):
    print(f'Error: {input_file} is not a valid lembox data file.\n')
    sys.exit(1)

print(f"Reading and updating file: {input_file}")

# Check raw data file for 'Samples' string. This indicates that the LEMBOX data recording function has injected a stdout 
# string into the data file (known issue). To resolve, we remove the lines before & after the bad line, + the bad line itself (minimal data loss)
try:
    with open(input_file, 'r') as f:
        rm = []
        r = f.readlines()

        for i, line in enumerate(r[1:]):
            numCommas = line.count(',')
            invalidCommas = numCommas != 6 and numCommas != 8
            if 'Sample' in line or invalidCommas:
                print(i)
                #print(line)
                rm.append(i + 1)

        #print(r[-5:])
        #print()

        rm.sort(reverse=True)
        for i in rm:

            
            #print(i - len(r))
            #print(r[i])

            r.pop(i)
            #if i + 1 <= len(r): r.pop(i + 1)
            #if i < len(r): r.pop(i)
            #if i - 1 < len(r): r.pop(i - 1)           #'Sample' string injection should not interrupt lines before first sighting of 'Sample' ?

        #print()
        #print(r[-5:])

    with open(input_file, 'w') as f:
        f.writelines(r)

except FileNotFoundError:
    print(f'{input_file} does not exist.')
    sys.exit(1)

    '''

df = pd.read_csv(input_file)
bad_rows = []
for i in range(len(df['Sample'])):
    try:
        int(df['Sample'])
        float(df['PerfTime(s)'])
        str(df['Timestamp'])
        str(df['VoltageRaw'])
        float(df['Voltage(V)'])
        str(df['CurrentRaw'])
        float(df['Current(A)'])
    except:
        bad_rows.append(i)

bad_rows.sort(reverse=True)
for r in bad_rows:
    df.drop(r)

dfToCsv(df, input_file)
'''

# added explicit dtype definitions to prevent dtypeWarning message. skipfooter avoids type error caused by debugging messages appended
# to data. python engine is required to use the skipfooter option.
col_types = {
    'Sample': np.int32,
    'PerfTime(s)': np.float64,
    'Timestamp': str,
    'VoltageRaw': str,
    'Voltage(V)': np.float64,
    'CurrentRaw': str,
    'Current(A)': np.float64,
}

# Read, scale, and update the data in place
if not csvHasColumn(input_file, 'Scaled_Voltage(V)'):
    df = pd.read_csv(input_file, dtype=col_types, skip_blank_lines=True, engine='python')
    df['Scaled_Voltage(V)'] = df['Voltage(V)'] * 10
    df['Scaled_Current(A)'] = df['Current(A)'] * 100

    # Save back to the same file
    df.to_csv(input_file, index=False)
    print(f"Scaling complete. Original file updated: {input_file}")
    
else:
    print(f'{input_file} already scaled!')