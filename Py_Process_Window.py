import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tkinter import filedialog

# Rosenthal model of weld pool to determin shape and prnitablity parameters. 
#Takes in csv file of physical properties and mesh size. 

def rosenthal_model(csv_filename):
    df = pd.read_csv(csv_filename)
   
   





def main():
    if len(sys.argv) != 2:
        csv_filename = filedialog.askopenfilename()
    else:
        csv_filename = sys.argv[1]

    if not csv_filename:
        print('No file selected.')
        return
    rosenthal_model(csv_filename)



if __name__ == "__main__":
    main()
