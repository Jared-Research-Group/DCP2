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

from datetime import datetime, timezone, timedelta

import cython

# Add build directory to path so compiled Cython modules can be found
build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'build', 'lib.win-amd64-cpython-310')
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

from data_manipulation import getRollingAvg, getRollingStdDev, getRollingSkew, getRollingKurtosis, getStartStop, dfHasColumn, dfAddColumn, dfToCsv, quickPlot
from lembox_scaling import scale_lembox

sample_rate = 20000 #hz

def getLemboxData(f, n = 1000, forceDataUpdate=False):
    print('         Reading LEMBOX data...')

    df = pd.read_csv(f)

    if not dfHasColumn(df, 'Scaled_Current(A)'):
        scale_lembox(os.path.split(f)[0])
        df = pd.read_csv(f)

    cdef double[:] curr = df['Scaled_Current(A)'].to_numpy()
    cdef double[:] volt = df['Scaled_Voltage(V)'].to_numpy()

    cdef double[:] avgV, avgI, time
    if not dfHasColumn(df, 'Avg_Voltage(V)') or forceDataUpdate or n==200:
        avgV =  getRollingAvg(volt, n)
        avgI = getRollingAvg(curr, n)

        count = df['Sample'].to_numpy().astype(float)
        time = count
        for r in range(len(time)):
            time[r] = time[r]/float(sample_rate)

        dfAddColumn(df, np.asarray(avgV), 'Avg_Voltage(V)')
        dfAddColumn(df, np.asarray(avgI), 'Avg_Current(A)')
        dfAddColumn(df, np.asarray(time), 'Interpolated_Time(s)')

        if n != 200:
            dfToCsv(df, f)
    else:
        avgV = df['Avg_Voltage(V)'].to_numpy()
        avgI = df['Avg_Current(A)'].to_numpy()
        time = df['Interpolated_Time(s)'].to_numpy()

    abs_time = []
    for t in df['Timestamp']:
        #abs_time.append(datetime.strptime((t[:-4] + ' UTC'), '%Y-%m-%d %H:%M:%S.%f %Z').astimezone(timezone(timedelta(hours=-10))).replace(tzinfo=None))        weird timezone issue (not present in all LEMBOX data?)
        abs_time.append(datetime.strptime((t[:-4]), '%Y-%m-%d %H:%M:%S.%f'))



    return  np.asarray(time), np.asarray(curr), np.asarray(volt), np.asarray(avgI), np.asarray(avgV), np.asarray(abs_time)

def drawStats(t, i, v, dir, startTime, stopTime, size, scale=sample_rate):

    if type(t) != list: t = t.tolist()
    if type(i) != list: i = i.tolist()
    if type(v) != list: v = v.tolist()

    df = pd.read_csv(dir + '/lembox_data.csv')
    if dfHasColumn(df, 'StdDev_Current(A)'):
        sd_i = df['StdDev_Current(A)'][startTime + scale - 1: stopTime]
        sd_v = df['StdDev_Voltage(V)'][startTime + scale - 1: stopTime]
        skew_i = df['Skew_Current'][startTime + scale - 1: stopTime]
        skew_v = df['Skew_Voltage'][startTime + scale - 1: stopTime]
        k_i = df['Kurtosis_Current'][startTime + scale - 1: stopTime]
        k_v = df['Kurtosis_Voltage'][startTime + scale - 1: stopTime]

    else:
        NaN_front = []
        NaN_back  = []
        for cnt in range(scale + startTime): NaN_front.append(float('NaN'))
        for cnt in range(size - stopTime - 1): NaN_back.append(float('NaN'))
            
        
        print('             Getting Rolling StdDevs...')
        sd_i = getRollingStdDev(i, scale)
        sd_v = getRollingStdDev(v, scale)

        dfAddColumn(df, NaN_front + sd_i + NaN_back, 'StdDev_Current(A)')
        dfAddColumn(df, NaN_front + sd_v + NaN_back, 'StdDev_Voltage(V)')

        print('             Getting Rolling Skews...')
        skew_i = getRollingSkew(i, scale, sd_i)
        skew_v = getRollingSkew(v, scale, sd_v)

        dfAddColumn(df, NaN_front + skew_i + NaN_back, 'Skew_Current')
        dfAddColumn(df, NaN_front + skew_v + NaN_back, 'Skew_Voltage')

        print('             Getting Rolling Kurtosis...')
        k_i = getRollingKurtosis(i, scale, sd_i)
        k_v = getRollingKurtosis(v, scale, sd_v)

        dfAddColumn(df, NaN_front + k_i + NaN_back, 'Kurtosis_Current')
        dfAddColumn(df, NaN_front + k_v + NaN_back, 'Kurtosis_Voltage')

        dfToCsv(df, dir + '/lembox_data.csv')

    t = t[scale-1:]
    i = i[scale-1:]
    v = v[scale-1:]

    data = [[v, sd_v, skew_v, k_v], [i, sd_i, skew_i, k_i]]
    label = [['Voltage (V)', 'Std Dev Voltage (V)', 'Skew Voltage', 'Kurtosis Voltage'], ['Current (A)', 'Std Dev Current (A)', 'Skew Current', 'Kurtosis Current']]

    fig, ax = plt.subplots(4,2, sharex=True)

    for e, typ in enumerate(data):
        for j, d in enumerate(typ):
            ax[j][e].scatter(t, d, s=0.005)

            if j > 1: ax[j][e].set_ylim(bottom = -10, top=10)

        for j, l in enumerate(label[e]):
            ax[j][e].set_ylabel(l)

    
    for a in ax[-1]: a.set_xlabel('Time (s)')
    fig.set_size_inches(30,10)
    fig.suptitle('Rolling Stats')
    plt.savefig(dir + '/visualizations/stats.png')
    
    print('             Stats Complete!')
    return

def plotHist(dir, v, i, t_start, f_prefix = ''):
    if f_prefix:
        if f_prefix[-1] != '_': f_prefix = f_prefix + '_'

    fig, ax = plt.subplots()
    ax.hist2d(v, i, bins=250, norm='log')
    fig.suptitle('Sample Statistical Distribution (' + str(len(v)) + ' data points starting from ' +str(t_start) +'s)')
    ax.set_xlabel('Voltage(V)')
    ax.set_ylabel('Current(A)')
    fig.set_size_inches(15,15)
    plt.savefig(dir + '/visualizations/' + f_prefix + 'histogram2D.png')
    
    fig, ax = plt.subplots(1, 2)
    ax[0].hist(i, bins=250)
    ax[0].set_xlabel('Current(A)')
    ax[1].hist(i, bins=250)
    ax[1].set_xlabel('Voltage(V)')
    
    fig.suptitle('Sample Statistical Distribution (' + str(len(v)) + ' data points starting from ' +str(t_start) +'s)')
    fig.set_size_inches(30,10)
    plt.savefig(dir + '/visualizations/' + f_prefix + 'histograms.png')
    return

def plotLemboxData(v, t, i, avgV, avgI, t_scale, file):
        fig, ax = plt.subplots(2,2, sharex=True)

        ax[0,0].scatter(t, v)
        ax[0,0].set_title('Voltage')
        ax[0,0].set_ylabel('Voltage (V)')

        ax[1,0].scatter(t, avgV)
        ax[1,0].set_title('Rolling Voltage Average Over ' + t_scale + ' Seconds')
        ax[1,0].set_ylabel('Average Voltage (V)')
        ax[1,0].set_xlabel('Time (s)')

        ax[0,1].scatter(t, i)
        ax[0,1].set_title('Current')
        ax[0,1].set_ylabel('Current (A)')

        ax[1,1].scatter(t, avgI)
        ax[1,1].set_title('Rolling Current Average Over ' + t_scale + ' Seconds')
        ax[1,1].set_ylabel('Average Current (A)')
        ax[1,1].set_xlabel('Time (s)')

        fig.set_size_inches(30,10)
        plt.savefig(file)

        return

def plotShortscaleLemboxData(v, t, i, file, n=25, p_start=None, p_stop=None):
        if type(t) == pd.Series: t.reset_index(inplace=True, drop=True)
        if type(v) == pd.Series: v.reset_index(inplace=True, drop=True)
        if type(i) == pd.Series: i.reset_index(inplace=True, drop=True)

        if not p_start: p_start=int(len(t)/2)
        if not p_stop:  p_stop= p_start + 20000*2

        fig, ax = plt.subplots(2,2, sharex=True)

        v = np.array(v)
        i = np.array(i)

        avgV = getRollingAvg(v, n)
        avgI = getRollingAvg(i, n)

        t_scale = t[int(len(t)/2) + n] - t[int(len(t)/2)]        # compute time length of rolling average
        t_scale = f"{t_scale:.5f}"

        ax[0,0].plot(t[p_start:p_stop], v[p_start:p_stop])
        ax[0,0].set_title('Voltage')
        ax[0,0].set_ylabel('Voltage (V)')

        ax[1,0].plot(t[p_start:p_stop], avgV[p_start:p_stop])
        ax[1,0].set_title('Rolling Voltage Average Over ' + t_scale + ' Seconds')
        ax[1,0].set_ylabel('Average Voltage (V)')
        ax[1,0].set_xlabel('Time (s)')

        ax[0,1].plot(t[p_start:p_stop], i[p_start:p_stop])
        ax[0,1].set_title('Current')
        ax[0,1].set_ylabel('Current (A)')

        ax[1,1].plot(t[p_start:p_stop], avgI[p_start:p_stop])
        ax[1,1].set_title('Rolling Current Average Over ' + t_scale + ' Seconds')
        ax[1,1].set_ylabel('Average Current (A)')
        ax[1,1].set_xlabel('Time (s)')

        fig.set_size_inches(30,10)
        plt.savefig(file)

        return

def drawLemboxVis(f, **kwargs):

    dir = os.path.split(f)[0]
    file = dir + '/visualizations/lembox.png'
    ss_file = dir + '/visualizations/shortscale_lembox.png'

    if 'pt_sz' in kwargs:
        pt_sz = kwargs['pt_sz']
    else:
        pt_sz = 0.005

    if 'avg_len' in kwargs:
        n = kwargs['avg_len']
    else:
        n = 1000

    t, i, v, avgI, avgV, timestamps = getLemboxData(f, n, False)

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

    begin = startTime + int(5* sample_rate)
    end = stopTime - int(5 * sample_rate)

    #drawStats(t[startTime:stopTime], i[startTime:stopTime], v[startTime:stopTime], dir, startTime, stopTime, len(t), 20000)
    plotLemboxData(v[startTime:stopTime], t[startTime:stopTime], i[startTime:stopTime], avgV[startTime:stopTime], avgI[startTime:stopTime], t_scale, file)
    plotHist(dir, v[begin:end], i[begin:end], t[begin])
    plotHist(dir, v[begin + int(5*sample_rate):begin + int(6*sample_rate)], i[begin + int(5*sample_rate):begin + int(6*sample_rate)], t[begin + int(5*sample_rate)], 'shortscale')

    mpl.rcParams['lines.markersize'] = 0.5
    plotShortscaleLemboxData(v[startTime:stopTime], t[startTime:stopTime], i[startTime:stopTime], ss_file)

    lemboxSpectrogram(v[startTime:stopTime], i[startTime:stopTime], dir)
    shortscaleFFT(v[begin:end], i[begin:end], f, sample_rate)

    print('             Average Voltage: ' + str(np.mean(v[begin:end])))
    print('             Average Current: ' + str(np.mean(i[begin:end])))
    print()

    for count in range(5):
        plt.close()
    getAvgStdDev(v[begin:end], i[begin:end], t[begin:end])
    
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
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(freq[1:int(len(freq)/5)], np.abs(iDFT[1:int(len(freq)/5)]), 'b', alpha=.85, label='Current FFT')
    ax[1].plot(freq[1:int(len(freq)/5)], np.abs(vDFT[1:int(len(freq)/5)]), 'r', alpha = .85, label='Voltage FFT')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('Voltage FFT Amplitude')
    ax[0].set_ylabel('Current FFT Amplitude')
    fig.set_size_inches(30,15)
    plt.savefig(dir + '/visualizations/lembox_fft.png')
    
    return

def lemboxSpectrogram(v, i, d, max_freq=1000):

    v = np.array(v)
    i = np.array(i)

    vFT = librosa.stft(v)
    V_db = librosa.amplitude_to_db(np.abs(vFT), ref=np.max)

    iFT = librosa.stft(i)
    I_db = librosa.amplitude_to_db(np.abs(iFT), ref=np.max)

    fig, ax = plt.subplots(layout='constrained')
    fig.set_size_inches(15,10)
    librosa.display.specshow(V_db, x_axis="time", y_axis="hz", sr=sample_rate)
    plt.ylim(top=max_freq)
    plt.colorbar()
    ax.set_title('Voltage')
    plt.savefig(d + '/visualizations/voltage_spectrogram.png')

    fig, ax = plt.subplots(layout='constrained')
    fig.set_size_inches(15,10)
    librosa.display.specshow(I_db, x_axis="time", y_axis="hz", sr=sample_rate)
    plt.ylim(top=max_freq)
    plt.colorbar()
    ax.set_title('Current')
    plt.savefig(d + '/visualizations/Current_spectrogram.png')

    return

def getAvgStdDev(v, i, t, plot=False, n=20000):
    v_roll = getRollingAvg(v, n)
    i_roll = getRollingAvg(i, n)

    print('             StdDev Voltage: ' + str(np.nanstd(v_roll)))
    print('             StdDev Current: ' + str(np.nanstd(i_roll)))

    if plot:
        fig, ax = plt.subplots(1, 2, layout='constrained')
        fig.set_size_inches(30,10)
        ax[0].plot(t, v_roll)
        ax[0].set_ylim([0, np.nanmax(v_roll) + 5])
        ax[1].plot(t, i_roll)
        ax[1].set_ylim([0, np.nanmax(i_roll) + 15])
        plt.show()

    return

def main(csv_file=None):
    if csv_file is not None:
        None
    elif len(sys.argv) != 2:
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