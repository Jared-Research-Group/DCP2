# Run these commands in the terminal
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python data_postprocessing.py

# data_postprocessing.py
The main function. It will prompt the user to select a build folder from an experiment. Then, it goes through the folder, seperates raw data from modified data, and executes functions to further process the data to more usable forms. The other Python scripts listed here are modules with functions to process different data types.
# audio_conversion.py
Converts audio recorded in CSV format to WAV format to enable listening

# lembox_scaling.py
Multiplies the voltage and current readings collected from the Miller LEM Box by 10 and 100 respectively to scale them to their true values

# robotdata_parsing.py
Takes robot messages recorded in text document with XML format and parses them, writing them to a CSV file

# create_flirvideo.py
Take FLIR data saved in numpy format and creates color mapped image frames and a video for viewing

# xiris.py
Build Xiris video (greyscale)

# thermocouple.py
Performs operations on thermocouple data.

# batch_process.py
Adds functionality for iterating over multiple data folders / build scale processing

# helper_functions.py
Collection of helpers for setup, data manipulation. Currently contains lots of things only used in mason-dev.