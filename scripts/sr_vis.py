import yaml
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.ticker as ticker
import numpy as np
import sympy as sp
from pathlib import Path

import helper_functions

def get_hist(yaml_file):

    with open(yaml_file, 'r') as f:
        metadata = yaml.unsafe_load(f)

    eq_hist   = metadata['equation_hist']
    loss_hist = metadata['loss_hist']
    
    return eq_hist, loss_hist

def plot_loss(loss, increment=1000):
    its = np.arange(len(loss)) * increment + increment

    fix, ax = plt.subplots(1, 1, layout='constrained')
    ax.plot(its, loss)
    ax.set_xlabel('PySR Iterations')
    ax.set_ylabel('Best-Fitting Function MSE Loss')
    ax.semilogy()
    ax.set_title('Low Temperature FLIR Calibration Loss History')

    plt.show()

def animate_eq_hist(dir, eq_hist, kelvin=False, symbolName='Pyrometer_Temperature', xlims=None, ylims=None):
    dir = Path(dir)

    fig, ax = plt.subplots(1, 1, layout='constrained')
    fig.set_size_inches(6, 4.5)

    x_val = np.arange(xlims[0], xlims[1], 1)
    x = sp.Symbol(symbolName)
    
    #plt.plot(x_val, sp.lambdify(x, eq_hist[-1])(x_val))
    #plt.show()
    
    formatter = ticker.FormatStrFormatter('%.2e')
    
    def drawNextFrame(eq_id):
        
        ax.clear()
        
        if xlims is not None:
            ax.set_xlim(xlims[0], xlims[1] + 50)
        if ylims is not None:
            ax.set_ylim(ylims[0], ylims[1])
            
        ax.grid(True)
        
        eqn = sp.lambdify(x, eq_hist[eq_id])
        
        if kelvin:
            ax.plot(x_val, eqn(x_val) - 273.15, c='blue')
        else:
            ax.plot(x_val, eqn(x_val), c='blue')
            
        ax.set_xlabel('Raw FLIR Intensity')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Low Temperature FLIR Calibration History (iteration %6d)' % (eq_id * 1000 + 1000))
        #ax.xaxis.set_major_formatter(formatter)

        return

    idx = range(len(eq_hist))

    ani = FuncAnimation(fig, drawNextFrame, frames=idx)
    ani.save(dir / 'equation_history.mp4', fps=10, dpi=600)
    
    print(eq_hist[-1])
    

if __name__ == '__main__':

    if len(sys.argv) == 2:
        filename = sys.argv[1]

    else:
        filename = helper_functions.selectFile()
        
    filename = Path(filename)

    eq_hist, loss_hist = get_hist(filename)
    
    plot_loss(loss_hist)

    pyro_xlims = (0, 1000)
    pyro_ylims = (-50, 500)

    flir_xlims = (0, 70000)
    flir_high_ylims = (-150, 1100)
    flir_low_ylims = (-50, 250)

    #animate_eq_hist(filename.parent, eq_hist, xlims=flir_xlims, ylims=flir_high_ylims, kelvin=True, symbolName='FLIR_Intensity')
    animate_eq_hist(filename.parent, eq_hist, kelvin=True, symbolName='FLIR_Intensity', xlims=flir_xlims, ylims=flir_low_ylims)