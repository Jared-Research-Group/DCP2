import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np

f = sys.argv[1]
df = pd.read_csv(f)

pos_x = df['RIst_X']
pos_y = df['RIst_Y']
pos_z = df['RIst_Z']
t     = df['RelativeTime']

plt.style.use('_mpl-gallery')
plt.rcParams['lines.markersize'] = 0.05
plt.rcParams['figure.constrained_layout.use'] = True

fig1, ax1 = plt.subplots(3,1, sharex=True)
ax1[0].scatter(t, pos_x)
ax1[1].scatter(t, pos_y)
ax1[2].scatter(t, pos_z)
fig1.set_size_inches(15,10)
ax1[0].set_title('Position - x')
ax1[1].set_title('Position - y')
ax1[2].set_title('Position - z')
ax1[2].set_xlabel('Time (s)')

fig3, ax3 = plt.subplots(subplot_kw={'projection': '3d'})
ax3.scatter(pos_x, pos_y, pos_z, s = 0.005)
ax3.set_aspect('equal')
ax3.disable_mouse_rotation()
fig3.set_size_inches(15,10)

vel_x = []
vel_y = []
vel_z = []
vel_comb = []

window = 1000
print(t[int(len(t)/2) + window] - t[int(len(t)/2)])

for p in range(len(pos_x[window:])):
    vel_x.append((pos_x[p+window] - pos_x[p])/(t[p+window] - t[p]))
    vel_y.append((pos_y[p+window] - pos_y[p])/(t[p+window] - t[p]))
    vel_z.append((pos_z[p+window] - pos_z[p])/(t[p+window] - t[p]))

fig2, ax2 = plt.subplots(3,1, sharex=True)
ax2[0].scatter(t[window:], vel_x)
ax2[1].scatter(t[window:], vel_y)
ax2[2].scatter(t[window:], vel_z)
fig2.set_size_inches(15,10)
ax2[0].set_title('Velocity - x')
ax2[1].set_title('Velocity - y')
ax2[2].set_title('Velocity - z')
ax2[2].set_xlabel('Time (s)')


scales = [0.005, 0.05, 0.25, 0.5, 0.75, 1, 1.5, 2]
fig4, ax4 = plt.subplots(len(scales), 1, sharex=True)

v_x = []
v_y = []
v_z = []
for i, scale in enumerate(scales):
    w = int(1000*scale)

    v_x.append([])
    v_y.append([])
    v_z.append([])
    for p in range(len(pos_x[w:])):
        v_x[-1].append((pos_x[p+w] - pos_x[p])/(t[p+w] - t[p]))
        v_y[-1].append((pos_y[p+w] - pos_y[p])/(t[p+w] - t[p]))
        v_z[-1].append((pos_z[p+w] - pos_z[p])/(t[p+w] - t[p]))

    ax4[i].scatter(t[w:], v_y[-1])
    ax4[i].set_title(f'V_y - window: {scale}')

ax4[-1].set_xlabel('Time (s)')
fig4.set_size_inches(15,10)

fig5, ax5 = plt.subplots()
ax5.scatter(t, np.gradient(pos_y, 500))
fig5.set_size_inches(15,10)

plt.show()


