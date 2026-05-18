import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
from tkinter             import filedialog

def Eccentricity_plot(csv_filename):
    df = pd.read_csv(csv_filename,parse_dates = ['Date Time'])
    Eccentricity = df['Eccentricity']
    Anisometry = df['Anisometry']
    Orientation = df['Orientation']
    Avg_Ecc = np.average(Eccentricity)
    print(Avg_Ecc)
    time = df['Date Time']
    plt.plot(time,Eccentricity,'o')
    plt.show()







def main():
    if len(sys.argv) != 2:
        csv_filename = filedialog.askopenfilename()
    else:
        csv_filename = sys.argv[1]
    Eccentricity_plot(csv_filename)



if __name__ == "__main__":
    main()
