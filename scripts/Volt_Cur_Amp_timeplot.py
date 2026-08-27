import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from tkinter             import filedialog

def plot_V_I_A(f,sample_time = 10, start_time = 10.5): #sample time in ms start time in seconds
    print(' ... Reading Aligned Data.')

    df = pd.read_csv(f)

    print('... Read Aligned Data.')
    sample_rate = 48000
    end_time = start_time + sample_time

    start_index = int(start_time*sample_rate)
    end_index = int(end_time*sample_rate)

    curr = df['Scaled_Current(A)'][start_index:end_index]
    volt = df['Scaled_Voltage(V)'][start_index:end_index]
    #Amp = df['Amplitude'][start_index:end_index]
    time = df['PerfTime(s)'][start_index:end_index]

    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(time, curr)
    axs[0].set_title('Current (A)')

    axs[1].plot(time, volt)
    axs[1].set_title('Voltage (V)')

    axs[2].plot(time, Amp)
    axs[2].set_title('Amplitude')

    """

    plt.scatter(time, curr, s=0.5)
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

    return 

def main():
    if len(sys.argv) != 2:
        csv_file = filedialog.askopenfilename()
    else:
        csv_file = sys.argv[1]

    plot_V_I_A(csv_file, sample_time=1, start_time=10)

if __name__ == "__main__":
    main()