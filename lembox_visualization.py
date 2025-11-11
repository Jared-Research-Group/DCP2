import matplotlib.pyplot as plt
import matplotlib        as mpl
import sys
import os
import pandas            as pd
import numpy             as np
import librosa
from tkinter             import filedialog
import subprocess
from pathlib import Path

from data_manipulation import getRollingAvg, getRollingStdDev, getRollingSkew, getRollingKurtosis, getStartStop, dfHasColumn, dfAddColumn, dfToCsv

sample_rate = 20000 #hz

def getLemboxData(f, n = 1000, forceDataUpdate=False):
    print('         Reading LEMBOX Data...')

    df = pd.read_csv(f)

    if not dfHasColumn(df, 'Scaled_Current(A)'):
        subprocess.run([sys.executable, str(Path(__file__).parent) + '/lembox_scaling.py', str(f)], check=True)
        df = pd.read_csv(f)

    curr = df['Scaled_Current(A)'].to_numpy()
    volt = df['Scaled_Voltage(V)'].to_numpy()

    if not dfHasColumn(df, 'Avg_Voltage(V)') or forceDataUpdate or n==200:
        avgV =  getRollingAvg(volt, n)
        avgI = getRollingAvg(curr, n)

        count = df['Sample'].to_numpy().astype(float)
        time = count
        for r in range(len(time)):
            time[r] = time[r]/float(sample_rate)

        dfAddColumn(df, avgV, 'Avg_Voltage(V)')
        dfAddColumn(df, avgI, 'Avg_Current(A)')
        dfAddColumn(df, time, 'Interpolated_Time(s)')

        if n != 200:
            dfToCsv(df, f)
    else:
        avgV = df['Avg_Voltage(V)']
        avgI = df['Avg_Current(A)']
        time = df['Interpolated_Time(s)']

    return  time, df['Scaled_Current(A)'], df['Scaled_Voltage(V)'], avgI, avgV

def drawStats(t, i, v, dir, scale=sample_rate):

    if type(t) != list: t = t.tolist()
    if type(i) != list: i = i.tolist()
    if type(v) != list: v = v.tolist()

    df = pd.read_csv(dir + '/lembox_data.csv')
    if dfHasColumn(df, 'StdDev_Current(A)'):
        sd_i = df['StdDev_Current(A)']
        sd_v = df['StdDev_Voltage(V)']
        skew_i = df['Skew_Current']
        skew_v = df['Skew_Voltage']
        k_i = df['Kurtosis_Current']
        k_v = df['Kurtosis_Voltage']

    else:
        print('             Getting Rolling StdDevs...')
        sd_i = getRollingStdDev(i, scale)
        sd_v = getRollingStdDev(v, scale)

        #dfAddColumn(df, sd_i, 'StdDev_Current(A)')
        #dfAddColumn(df, sd_v, 'StdDev_Voltage(V)')

        print('             Getting Rolling Skews...')
        skew_i = getRollingSkew(i, scale, sd_i)
        skew_v = getRollingSkew(v, scale, sd_v)

        #dfAddColumn(df, skew_i, 'Skew_Current')
        #dfAddColumn(df, skew_v, 'Skew_Voltage')

        print('             Getting Rolling Kurtosis...')
        k_i = getRollingKurtosis(i, scale, sd_i)
        k_v = getRollingKurtosis(v, scale, sd_v)

        #dfAddColumn(df, k_i, 'Kurtosis_Current')
        #dfAddColumn(df, k_v, 'Kurtosis_Voltage')

        #dfToCsv(df, dir + '/lembox_data.csv')

    t = t[scale-1:]
    i = i[scale-1:]
    v = v[scale-1:]

    data = [[v, sd_v, skew_v, k_v], [i, sd_i, skew_i, k_i]]
    label = [['Voltage (V)', 'Std Dev Voltage (V)', 'Skew Voltage', 'Kurtosis Voltage'], ['Current (A)', 'Std Dev Current (A)', 'Skew Current', 'Kurtosis Current']]

    fig, ax = plt.subplots(4,2, sharex=True)

    for i, typ in enumerate(data):
        for j, d in enumerate(typ):
            ax[j][i].scatter(t, d, s=0.005)

            if j > 1: ax[j][i].set_ylim(bottom = -10, top=10)

        for j, l in enumerate(label[i]):
            ax[j][i].set_ylabel(l)

    
    for a in ax[-1]: a.set_xlabel('Time (s)')
    fig.set_size_inches(30,10)
    fig.suptitle('Rolling Stats')
    #plt.show()
    plt.savefig(dir + '/visualizations/stats.png')

    print('             Stats Complete!')

    return

def plotLemboxData(v, t, i, avgV, avgI, t_scale, file, p_start, p_stop):
        fig, ax = plt.subplots(2,2, sharex=True)

        ax[0,0].scatter(t[p_start:p_stop], v[p_start:p_stop])
        ax[0,0].set_title('Voltage')
        ax[0,0].set_ylabel('Voltage (V)')

        ax[1,0].scatter(t[p_start:p_stop], avgV[p_start:p_stop])
        ax[1,0].set_title('Rolling Voltage Average Over ' + t_scale + ' Seconds')
        ax[1,0].set_ylabel('Average Voltage (V)')
        ax[1,0].set_xlabel('Time (s)')

        ax[0,1].scatter(t[p_start:p_stop], i[p_start:p_stop])
        ax[0,1].set_title('Current')
        ax[0,1].set_ylabel('Current (A)')

        ax[1,1].scatter(t[p_start:p_stop], avgI[p_start:p_stop])
        ax[1,1].set_title('Rolling Current Average Over ' + t_scale + ' Seconds')
        ax[1,1].set_ylabel('Average Current (A)')
        ax[1,1].set_xlabel('Time (s)')

        fig.set_size_inches(30,10)
        plt.savefig(file)

        return

def drawLemboxVis(f, **kwargs):

    dir = os.path.split(f)[0]
    file = dir + '/visualizations/lembox.png'

    startup = False
    p_start = 0
    p_stop = -1

    if 'pt_sz' in kwargs:
        pt_sz = kwargs['pt_sz']
    else:
        pt_sz = 0.005

    if 'avg_len' in kwargs:
        n = kwargs['avg_len']
    else:
        n = 1000

    # this should call a second call to drawLemboxVis
    if 'startup' in kwargs:
        if kwargs['startup'] == True:
            pt_sz = 0.25
            n = 200
            file = dir + '/visualizations/shortscale_lembox.png'
            startup = True
            #p_start = int(1.95*sample_rate)
            #p_stop = int(2*sample_rate) + int(0.5*sample_rate) + int(0.1*sample_rate)
            p_start = int(10*sample_rate)
            p_stop = int(10*sample_rate) + int(0.6*sample_rate)


    t, i, v, avgI, avgV = getLemboxData(f, n)

    t_scale = t[int(len(t)/2) + n] - t[int(len(t)/2)]        # compute time length of rolling average
    t_scale = f"{t_scale:.5f}"

    startTime, stopTime = getStartStop(avgV, 1)

    startTime -= 2*sample_rate      # set observed window to welding time +- 2 seconds
    stopTime += 2*sample_rate
    if startTime < 0:
        startTime = 0
    if stopTime > len(avgV):
        stopTime = len(avgV)

    st, so = getStartStop(v, 1)

    st -= 20000
    so -= 20000

    print('         Drawing LEMBOX vis...')

    plt.style.use('_mpl-gallery')
    mpl.rcParams['lines.markersize'] = pt_sz*2
    mpl.rcParams['figure.constrained_layout.use'] = True

    begin = 5* sample_rate
    l = int(10 * sample_rate)
    shortscaleFFT(v[begin:(begin+l)], i[begin:(begin+l)], f, sample_rate)

    drawStats(t[startTime:stopTime], i[startTime:stopTime], v[startTime:stopTime], dir, 20000)
    plotLemboxData(v[startTime:stopTime], t[startTime:stopTime], i[startTime:stopTime], avgV[startTime:stopTime], avgI[startTime:stopTime], t_scale, file, p_start, p_stop)
    
    return

def shortscaleFFT(v, i, f, rate):
    vArr = np.array(v)
    iArr = np.array(i)

    dir = os.path.split(f)[0]

    vDFT = np.fft.rfft(vArr)
    iDFT = np.fft.rfft(iArr)

    freq = np.fft.rfftfreq(len(v), 1/rate)

    #mpl.rcParams['lines.markersize'] = 0.05*2
    plt.style.use('_mpl-gallery')
    fig, ax = plt.subplots(2,1)
    ax[0].plot(freq[1:int(len(freq)/50)], np.abs(iDFT[1:int(len(freq)/50)]), 'r', alpha=.85, label='Current FFT')
    ax[1].plot(freq[1:int(len(freq)/50)], np.abs(vDFT[1:int(len(freq)/50)]), 'b', alpha = .85, label='Voltage FFT')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('Voltage FFT Amplitude')
    ax[0].set_ylabel('Current FFT Amplitude')
    fig.set_size_inches(15,20)
    plt.savefig(dir + '/visualizations/lembox_fft.png')
    
    return

def lemboxSpectrogram(v, i, t, d):

    return

def main():
    if len(sys.argv) != 2:
        csv_file = filedialog.askopenfilename()
    else:
        csv_file = sys.argv[1]
    
    dir = os.path.split(csv_file)[0]

    try:
        os.mkdir(dir + '/visualizations')
    except FileExistsError:
        pass

    print('Generating LEMBOX data visualizations...')
    drawLemboxVis(csv_file, startup=False)
    print('LEMBOX visualization complete')
    return

if __name__ == "__main__":
    main()