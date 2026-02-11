import os
import sys
import pandas as pd
import numpy as np
import wave

def csv_to_wav(csv_filename, wav_filename=None, sampling_rate=48000):

    print(f"Reading {csv_filename}...")
    try:
        # Try reading with one header row skipped (in case there's extra info)
        df = pd.read_csv(csv_filename, skiprows=1, low_memory=False)
        print(f"Columns found (skiprows=1): {df.columns.tolist()}")
        if 'Amplitude' not in df.columns:
            # If 'Amplitude' isn't found, try reading the CSV normally.
            df = pd.read_csv(csv_filename, low_memory=False)
            print(f"Retrying with all rows: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if 'Amplitude' not in df.columns:
        print(f"Column 'Amplitude' not found in {csv_filename}. Skipping...")
        return

    try:
        # Extract mono audio data from the 'Amplitude' column.
        audio_data = df['Amplitude'].to_numpy()
        
        print(f"\nDebug Information for {csv_filename}:")
        print(f"Raw audio data range: {audio_data.min():.6f} to {audio_data.max():.6f}")
        print(f"Number of samples: {len(audio_data)}")
        
        # Normalize the signal: scale so maximum amplitude is 0.9.
        max_amplitude = np.max(np.abs(audio_data))
        if max_amplitude > 0:
            audio_data = audio_data / max_amplitude * 0.9
        
        # Convert the normalized data to 16-bit integers.
        audio_data_int16 = (audio_data * 32767).astype(np.int16)
        
        # Generate output filename if not provided.
        if wav_filename is None:
            wav_filename = os.path.splitext(csv_filename)[0] + '.wav'
        
        # Save the WAV file using Python's wave module.
        print(f"\nSaving to {wav_filename}...")
        with wave.open(wav_filename, 'wb') as wav_file:
            wav_file.setnchannels(1)   # Mono
            wav_file.setsampwidth(2)    # 2 bytes per sample for 16-bit
            wav_file.setframerate(sampling_rate)
            wav_file.writeframes(audio_data_int16.tobytes())
        
        print(f"WAV file saved successfully as {wav_filename}\n")
        
    except Exception as e:
        print(f"Error during conversion for {csv_filename}: {e}")
        return

def main():
    if len(sys.argv) != 2:
        print("Usage: python audio_conversion.py <input_csv_file>")
        sys.exit(1)
    
    # Get input file path from command line argument
    csv_file = sys.argv[1]
    
    # Generate output WAV filename in same directory as input
    output_wav = os.path.splitext(csv_file)[0] + '.wav'
    
    # Process the file
    csv_to_wav(csv_file, wav_filename=output_wav, sampling_rate=48000)

if __name__ == "__main__":
    main()