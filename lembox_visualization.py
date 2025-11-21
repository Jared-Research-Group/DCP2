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

from data_manipulation import getRollingAvg, getRollingStdDev, getRollingSkew, getRollingKurtosis, getStartStop, dfHasColumn, dfAddColumn, dfToCsv, quickPlot

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
    #plt.show()
    plt.savefig(dir + '/visualizations/stats.png')
    
    sample_t = t[int(len(t)/2): int(len(t)/2) + scale]
    sample_i = i[int(len(t)/2): int(len(t)/2) + scale]
    sample_v = v[int(len(t)/2): int(len(t)/2) + scale]
    
    fig, ax = plt.subplots()
    ax.hist2d(i, v, bins=100, norm='log')
    fig.suptitle('Sample Statistical Distribution')
    ax.set_xlabel('Current(A)')
    ax.set_ylabel('Voltage(V)')
    fig.set_size_inches(15,15)
    plt.savefig(dir + '/visualizations/histogram2D.png')
    
    fig, ax = plt.subplots(1, 2)
    ax[0].hist(i, bins=250)
    ax[0].set_xlabel('Current(A)')
    ax[1].hist(v, bins=250)
    ax[1].set_xlabel('Voltage(V)')
    
    fig.suptitle('Sample Statistical Distribution')
    fig.set_size_inches(30,10)
    plt.savefig(dir + '/visualizations/histograms.png')
    
    print('             Stats Complete!')
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

def plotShortscaleLemboxData(v, t, i, file, n=25, p_start=20000*5, p_stop=20000*7):
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

    t, i, v, avgI, avgV, globalStart = getLemboxData(f, n, False)

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

    begin = startTime + int(7* sample_rate)
    end = stopTime - int(5 * sample_rate)

    #drawStats(t[startTime:stopTime], i[startTime:stopTime], v[startTime:stopTime], dir, 20000)
    plotLemboxData(v[startTime:stopTime], t[startTime:stopTime], i[startTime:stopTime], avgV[startTime:stopTime], avgI[startTime:stopTime], t_scale, file)
    mpl.rcParams['lines.markersize'] = 0.5
    plotLemboxData(v[int(22.5*20000):int(32.5*20000)], t[int(22.5*20000):int(32.5*20000)], i[int(22.5*20000):int(32.5*20000)], avgV[int(22.5*20000):int(32.5*20000)], avgI[int(22.5*20000):int(32.5*20000)], t_scale, dir + '/visualizations/lembox_defect.png')
    plotLemboxData(v[int(22.5*20000):int(30*20000)], t[int(22.5*20000):int(30*20000)], i[int(22.5*20000):int(30*20000)], avgV[int(22.5*20000):int(30*20000)], avgI[int(22.5*20000):int(30*20000)], t_scale, dir + '/visualizations/lembox_defect_2.png')
    plotLemboxData(v[int(30*20000):int(32.5*20000)], t[int(30*20000):int(32.5*20000)], i[int(30*20000):int(32.5*20000)], avgV[int(30*20000):int(32.5*20000)], avgI[int(30*20000):int(32.5*20000)], t_scale, dir + '/visualizations/lembox_defect_3.png')
    plotShortscaleLemboxData(v[startTime:stopTime], t[startTime:stopTime], i[startTime:stopTime], ss_file)
    lemboxSpectrogram(v[startTime:stopTime], i[startTime:stopTime], dir)

    shortscaleFFT(v[begin:end], i[begin:end], f, sample_rate)

    #quickPlot((t[begin:begin+(20*70)], avgV[begin:begin+(20*70)]), (t[begin:begin+(20*70)], avgI[begin:begin+(20*70)]))

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

def lemboxSpectrogram(v, i, d):

    v = np.array(v)
    i = np.array(i)

    vFT = librosa.stft(v)
    V_db = librosa.amplitude_to_db(np.abs(vFT), ref=np.max)

    iFT = librosa.stft(i)
    I_db = librosa.amplitude_to_db(np.abs(iFT), ref=np.max)

    fig, ax = plt.subplots(layout='constrained')
    fig.set_size_inches(15,10)
    librosa.display.specshow(V_db, x_axis="time", y_axis="hz", sr=sample_rate)
    plt.colorbar()
    ax.set_title('Voltage')
    plt.savefig(d + '/visualizations/voltage_spectrogram.png')

    fig, ax = plt.subplots(layout='constrained')
    fig.set_size_inches(15,10)
    librosa.display.specshow(I_db, x_axis="time", y_axis="hz", sr=sample_rate)
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