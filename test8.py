import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as colors
import matplotlib.cm     as cm
import numpy as np
import pysr
import os
import sympy as sp

import helper_functions



img = np.load(r"C:\Users\w2w\Data\rebase\modified_data\data_collection_20251211_162617\FLIR\FLIR-Frame-690.npy", allow_pickle=True).item()['frame']

high_fit = pysr.PySRRegressor().from_file(run_directory=os.getcwd() + '/FLIR_fits/High', model_selection='best', verbosity=0)
x = sp.Symbol('FLIR_Intensity')

print(high_fit.sympy(11))
model    = sp.lambdify(x, high_fit.sympy(11), modules='numpy')

img = helper_functions.flirConversion(img, model)

temp_max = helper_functions.flirConversion(2**16 - 1, model)

fig, ax = plt.subplots(1, 1)
ax.imshow(img, cmap='inferno', vmin=np.nanmin(img) - 10, vmax=temp_max)
ax.axis('off')

fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
ax.margins(0)

axins = inset_axes(ax, width="5%", height="50%", loc='upper right')
axins.yaxis.set_label_position('left')
axins.tick_params(colors='white')
cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=np.nanmin(img), vmax=np.nanmax(img)), cmap='inferno'), cax=axins, ticklocation='left')

#cbar.set_label('IR Camera Intensity', color='white')
cbar.set_label('Calibrated Temperature (°C)', color='white')

plt.show()
fig.savefig(r"C:\Users\w2w\Downloads\sample_flir.png")