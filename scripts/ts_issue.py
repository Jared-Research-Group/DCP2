import pandas as pd
import matplotlib.pyplot as plt
import datetime
import copy

from data_manipulation import getStartStop

#print('reading mic...')
#mic_df = pd.read_csv("C:/Users/wwerner4/Desktop/MSE 525/Bead 2/data_collection_20251120_120204/microphone_data.csv", parse_dates=['Absolute Time'], skiprows=1)
print('reading lem...')
lem_df = pd.read_csv("C:/Users/wwerner4/Desktop/MSE 525/Bead 4/data_collection_20251120_121145/lembox_data.csv")

lem_df['Timestamp'] = lem_df['Timestamp'].apply(lambda t: t[:-4])
lem_df['Timestamp'] = lem_df['Timestamp'].apply(datetime.datetime.strptime, args=('%Y-%m-%d %H:%M:%S.%f',))

time0  = lem_df.loc[0, 'Timestamp']
abs_times = copy.copy(lem_df['Timestamp'])
abs_times = abs_times.apply(lambda t: t - time0)

time0 = lem_df.loc[0, 'PerfTime(s)']
rel_times = copy.copy(lem_df['PerfTime(s)'])
rel_times = rel_times.apply(lambda t: t - time0)

print(lem_df.loc[915904, 'PerfTime(s)'])
print(lem_df.loc[915904, 'Timestamp'])

bad_rel = []
bad_abs = []
diff = []
for i, row in enumerate(zip(abs_times, rel_times)):
    abs, rel = row
    time_diff = (abs - datetime.timedelta(seconds=rel)).total_seconds()

    if time_diff > 0.05:
        bad_rel.append(lem_df.loc[i, 'PerfTime(s)'])
        bad_abs.append(lem_df.loc[i, 'Timestamp'])
        diff.append(time_diff)

bad_times = pd.DataFrame({'rel_time':bad_rel, 'abs_time':bad_abs, 'diff':diff})

print(bad_times)
        

'''
print('plotting')
start_lem, stop_lem = getStartStop(lem_df['Scaled_Voltage(V)'])

start_mic = int(start_lem/20000 * 48000)
stop_mic = int(stop_lem/20000 * 48000)

fig, ax = plt.subplots(2,1, layout='constrained')
ax[0].plot(mic_df['Absolute Time'][start_mic + 48000*3:start_mic + 48000*4], mic_df['Amplitude'][start_mic + 48000*3:start_mic + 48000*4])
ax[1].plot(mic_df['Relative Time (s)'][start_mic + 48000*3:start_mic + 48000*4], mic_df['Amplitude'][start_mic + 48000*3:start_mic + 48000*4])

plt.show()

'''