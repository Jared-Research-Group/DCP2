import os
import sys
from pathlib import Path
import pandas as pd

import xml.etree.ElementTree as ET

from core_scripts.helper_functions import selectFolder

def extract_xml_data(element, prefix=''):
    """Recursively extract all data from XML elements"""
    data = {}
    
    # Handle attributes (like X, Y, Z in RIst/RSol)
    for key, value in element.attrib.items():
        try:
            # Try to convert to float or int, fall back to string if fails
            data[f"{prefix}{element.tag}_{key}"] = float(value)
        except ValueError:
            data[f"{prefix}{element.tag}_{key}"] = value
    
    # Handle element text (like in CAM, FLASH)
    if element.text and element.text.strip():
        try:
            data[element.tag] = float(element.text)
        except ValueError:
            data[element.tag] = element.text.strip()
    
    # Recursively process child elements
    for child in element:
        data.update(extract_xml_data(child, prefix))
    
    return data

def parse_robot_message(line):
    # Split the line into timestamp and XML parts
    parts = line.strip().split('|')
    if len(parts) != 3:
        return None
    
    system_time, relative_time, xml_data = parts
    
    try:
        # Parse XML data
        root = ET.fromstring(xml_data)
        
        # Extract all XML data dynamically
        parsed_data = extract_xml_data(root)
        
        # Add timestamp data
        parsed_data.update({
            'SystemTime': system_time,
            'RelativeTime': float(relative_time)
        })
        
        return parsed_data
        
    except Exception as e:
        print(f"Error parsing line: {e}")
        return None

# TODO: add kwarg functionality to modify default filenames
def convert_robot_data_to_csv(dir):

    """ 
        Convert raw robot data *.txt file to *.csv
    """

    dir = Path(dir)
    if not os.access(dir / 'raw_data', os.R_OK):
        os.mkdir(dir / 'raw_data')

    output_file = dir / 'robot_data.csv'
    input_file = dir / 'raw_data' / 'robot_data__raw.txt'
    
    os.replace(dir / 'robot_data.txt', input_file)

    data_list = []
    
    # Read and parse the input file
    with open(input_file, 'r') as f:
        # Skip the header line
        next(f)
        
        for line in f:
            if line.strip():  # Skip empty lines
                parsed_data = parse_robot_message(line)
                if parsed_data:
                    data_list.append(parsed_data)
    
    # Convert to DataFrame and save to CSV
    if data_list:
        df = pd.DataFrame(data_list)
        
        # Define timestamp columns to appear first
        timestamp_cols = ['SystemTime', 'RelativeTime']
        
        # Get all other columns
        other_cols = [col for col in df.columns if col not in timestamp_cols]
        
        # Reorder columns
        df = df[timestamp_cols + other_cols]
        
        df.to_csv(output_file, index=False)
        print(f"Data successfully written to {output_file}")

    else:
        print("No data was parsed")

if __name__ == "__main__":
        
        if len(sys.argv) == 2:
            dir = sys.argv[1]

        else:
            dir = selectFolder()

        convert_robot_data_to_csv(dir)