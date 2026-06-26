import pandas as pd
import numpy as np
import wave

import helper_functions

logger = helper_functions.setup_logger(__name__)

def csv_to_wav(dir, sampling_rate=48000, save_cols=['Absolute Time', 'Amplitude'], **kwargs):

    """
        Convert raw microphone *.csv to *.wav file
    """

    # define default input/output file names
    input_file = 'microphone_data.csv'
    output_files = ['microphone_data.wav', 'microphone_data__clean.csv']

    # get current data path, paths to store new data
    [csv_filename, [wav_filename, new_csv]] = helper_functions.setup_directory_structure(dir, input_file, output_files, **kwargs)

    logger.info(f"Reading {csv_filename}...")
    try:
        # Try reading with one header row skipped (in case there's extra info)
        df = pd.read_csv(csv_filename, skiprows=1, low_memory=False)
        logger.debug(f"Columns found (skiprows=1): {df.columns.tolist()}")

        if 'Amplitude' not in df.columns:
            # If 'Amplitude' isn't found, try reading the CSV normally.
            df = pd.read_csv(csv_filename, low_memory=False)
            logger.debug(f"Retrying with all rows: {df.columns.tolist()}")

    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return

    if 'Amplitude' not in df.columns:
        logger.error(f"Column 'Amplitude' not found in {csv_filename}. Skipping...")
        return

    try:
        # Extract mono audio data from the 'Amplitude' column.
        audio_data = df['Amplitude'].to_numpy()
        
        logger.debug(f"\nDebug Information for {csv_filename}:\nRaw audio data range: {audio_data.min():.6f} to {audio_data.max():.6f}\nNumber of samples: {len(audio_data)}")
        
        # Normalize the signal: scale so maximum amplitude is 0.9.
        max_amplitude = np.max(np.abs(audio_data))
        if max_amplitude > 0:
            audio_data = audio_data / max_amplitude * 0.9
        
        # Convert the normalized data to 16-bit integers.
        audio_data_int16 = (audio_data * (2**15 - 1)).astype(np.int16)
        
        # Save the WAV file using Python's wave module.
        logger.info(f"Saving to {wav_filename}...")
        with wave.open(str(wav_filename), 'wb') as wav_file:
            wav_file.setnchannels(1)   # Mono
            wav_file.setsampwidth(2)    # 2 bytes per sample for 16-bit
            wav_file.setframerate(sampling_rate)
            wav_file.writeframes(audio_data_int16.tobytes())
        
        logger.info(f"WAV file saved successfully as {wav_filename}")
        
    except Exception as e:
        logger.error(f"Error during conversion for {csv_filename}: {e}")
        return
    
    df[save_cols].to_csv(new_csv, index=False)

if __name__ == '__main__':

    [dir, kwargs] = helper_functions.setup_kwargs(__name__, 2)

    csv_to_wav(dir, **kwargs)
    