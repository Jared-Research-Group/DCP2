import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from tkinter             import filedialog
import pywt
from denoising import WaveletDenoising

def WDN(f):
    print('...DWT of Amplitude Data')
    df = pd.read_csv(f)
    Amp = df['Amplitude']
    Time = df['time']

    wd = WaveletDenoising(normalize=False,
                      wavelet='db3',
                      level=10,
                      thr_mode='hard',
                      selected_level=None,
                      method="universal",
                      energy_perc=0.90)
   
    denoised_Amp = wd.fit(Amp)

    fig = plt.figure()
    plt.plot(Time, Amp, 'blue')
    plt.plot(Time, denoised_Amp,'red')
    plt.show()

def main():
    if len(sys.argv) != 2:
        csv_file = filedialog.askopenfilename()
    else:
        csv_file = sys.argv[1]
    
    wavelet = WDN(csv_file)
  
if __name__ == "__main__":
    main()