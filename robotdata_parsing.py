import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

import helper_functions

logger = helper_functions.setup_logger(__name__)

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
        logger.error(f"Error parsing line: {e}")
        return None

def convert_robot_data_to_csv(dir, **kwargs):

    """ 
        Convert raw robot data *.txt file to *.csv
    """

    input_filename = 'robot_data.txt'
    output_filenames = ['robot_data__clean.csv']
    
    [input_file, [output_file]] = helper_functions.setup_directory_structure(dir, input_filename, output_filenames, **kwargs)

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

        [IpocUnique, IpocIndex] = np.unique(df['IPOC'], return_index=True)

        ipoc_start = IpocUnique[0]
        time_steps = [timedelta(milliseconds= ipoc - ipoc_start) for ipoc in IpocUnique]

        df = df.iloc[IpocIndex]
        df['SystemTime'] = [step + datetime.strptime(df['SystemTime'][0], '%Y-%m-%d %H:%M:%S.%f') for step in time_steps]
        
        timestamp_cols = ['SystemTime', 'RelativeTime']
        
        # Get all other columns
        other_cols = [col for col in df.columns if col not in timestamp_cols]
        
        # Reorder columns
        df = df[['SystemTime'] + other_cols]
        
        df.to_csv(output_file, index=False)
        logger.info(f"Data successfully written to {output_file}")

    else:
        logger.warning("No data was parsed")

if __name__ == "__main__":
        
        [dir, kwargs] = helper_functions.setup_kwargs(__name__, 1)

        convert_robot_data_to_csv(dir, **kwargs)