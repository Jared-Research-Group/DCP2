import pandas as pd
from datetime import datetime

import xml.etree.ElementTree as ET

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

def convert_robot_data_to_csv(input_file, output_file):
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
    import sys
    import os
    
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if len(sys.argv) == 3:
        # Make paths relative to script directory if they're not absolute
        input_file = os.path.join(script_dir, sys.argv[1]) if not os.path.isabs(sys.argv[1]) else sys.argv[1]
        output_file = os.path.join(script_dir, sys.argv[2]) if not os.path.isabs(sys.argv[2]) else sys.argv[2]
    else:
        input_file = os.path.join(script_dir, "robot_data.txt")
        output_file = os.path.join(script_dir, "robot_data_parsed.csv")
    
    # Convert paths to proper format
    input_file = os.path.normpath(input_file)
    output_file = os.path.normpath(output_file)
    
    print(f"Attempting to read from: {input_file}")
    convert_robot_data_to_csv(input_file, output_file)