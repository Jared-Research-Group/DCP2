import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd

def drawAudioVis(wavFile):
    dir = os.path.split(wavFile)[0]

    sr = librosa.get_samplerate(wavFile)
    array, sr = librosa.load(wavFile, sr=sr)

    #array = np.array(array)
    csv_filename = dir + '/microphone_data.csv'
    df = pd.read_csv(csv_filename, skiprows=1, low_memory=False)
    array = df['Amplitude'].to_numpy()

    fig, ax = plt.subplots(layout='constrained')
    fig.set_size_inches(15,10)
    #plt.rcParams['lines.linewidth'] = 0.0001       # trying to make waveform readable
    librosa.display.waveshow(array, sr=sr)
    plt.savefig(dir + '/visualizations/audio_waveform.png')

    D = librosa.stft(array)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    fig, ax = plt.subplots(layout='constrained')
    fig.set_size_inches(15,10)
    librosa.display.specshow(S_db, x_axis="time", y_axis="hz", sr=sr, fmax=2000)
    plt.colorbar()
    plt.savefig(dir + '/visualizations/audio_spectrogram.png')


    arrDFT = np.fft.rfft(array)
    freq = np.fft.rfftfreq(len(array), 1/sr)

    #mpl.rcParams['lines.markersize'] = 0.05*2
    plt.style.use('_mpl-gallery')
    fig, ax = plt.subplots()
    ax.plot(freq[1:int(len(freq)/25 * 2)], np.abs(arrDFT[1:int(len(freq)/25 * 2)]), 'b', alpha=.85, label='Current FFT')
    fig.set_size_inches(15,15)
    plt.savefig(dir + '/visualizations/audio_fft.png')

    return

def main():
    if len(sys.argv) != 2:
        print("Usage: python audio_visualization.py <input_wav_file>")
        sys.exit(1)
    
    # Get input file path from command line argument
    wavFile = sys.argv[1]
    dir = os.path.split(wavFile)[0]

    try:
        os.mkdir(dir + '/visualizations')
    except FileExistsError:
        pass

    print('Generating microphone data visualizations...')
    drawAudioVis(wavFile)
    print('Audio visualization complete')
    return

if __name__ == "__main__":
    main()