import yaml
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
    its = np.arange(len(loss)) * increment

    plt.tight_layout()
    plt.plot(its, loss)
    plt.xlabel('PySR Iterations')
    plt.ylabel('Best-Fitting Function MSE Loss')
    plt.semilogy()

    plt.show()

def animate_eq_hist(dir, eq_hist, xlims=None, ylims=None):
    dir = Path(dir)

    fig, ax = plt.subplots(1, 1, layout='constrained')

    if xlims is not None:
        ax.set_xlim(xlims[0], xlims[1] + 50)
    if ylims is not None:
        ax.set_ylim(ylims[0] - 50, ylims[1] + 50)

    x_val = np.arange(xlims[0], xlims[1], 10)
    x = sp.Symbol('FLIR_Intensity')
    def drawNextFrame(eq_id):
        eqn = sp.lambdify(x, eq_hist[eq_id])
        ax.plot(x_val, eqn(x_val) - 273.15)

        return

    idx = range(len(eq_hist))

    ani = FuncAnimation(fig, drawNextFrame, frames=idx)
    ani.save(dir / 'equation_history.mp4', fps=15)
    

if __name__ == '__main__':

    if len(sys.argv) == 2:
        filename = sys.argv[1]

    else:
        filename = helper_functions.selectFile()
        
    filename = Path(filename)

    eq_hist, loss_hist = get_hist(filename)
    
    plot_loss(loss_hist)

    pyro_xlims = (-45, 1370)
    pyro_ylims = (-45, 1370)

    flir_xlims = (0, 2**16)
    flir_high_ylims = (-100, 1050)
    flir_low_ylims = (-200, 200)

    animate_eq_hist(filename.parent, eq_hist, flir_xlims, flir_high_ylims)