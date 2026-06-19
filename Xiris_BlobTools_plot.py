import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
from tkinter             import filedialog

def Xiris_data_plot(csv_filename):
    df = pd.read_csv(csv_filename,parse_dates = ['Date Time'])
    Moment2x = df['Moment2x']
    print(Moment2x)
    Orientation = df['Orientation']
    time = df['Date Time']
    plt.plot(time,Moment2x,'o')
    plt.show()



def main():
    if len(sys.argv) != 2:
        csv_filename = filedialog.askopenfilename()
    else:
        csv_filename = sys.argv[1]
    Xiris_data_plot(csv_filename)



if __name__ == "__main__":
    main()
