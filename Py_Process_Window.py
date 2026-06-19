import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from tkinter import filedialog

# Rosenthal model of weld pool to determin shape and prnitablity parameters. 
#Takes in csv file of physical properties and mesh size. 

def rosenthal_model(csv_filename):

    # Read input parameters from the Excel file
    df = pd.read_excel(csv_filename,sheet_name='Input')
    Material = df['Material'][0] # Material being welded
    print(f'Material: {Material}')
    B_T = df['Base Temperature [°C]'][0] # Base temperature in degrees Celsius 
    k   = df['Thermal Conductivity [W/mK]'][0] # Thermal conductivity in W/mK
    rho = df['Density [kg/m^3]'][0] # Density in kg/m^3
    c   = df['Specific Heat Capacity [J/kgK]'][0] # Specific heat capacity in J/kgK
    eta = df['Efficiency'][0] # Efficiency of the welding process
    L   = df['liquidus temperature [°C]'][0] # Liquidus temperature in degrees Celsius
    S   = df['Solidus temperature [°C]'][0] # Solidus temperature in degrees Celsius
    v   = np.arange(df['Min v [m/s]'][0], df['Max v [m/s]'][0]+df['v step'][0], df['v step'][0]) # Welding speed in m/s
    w_d = df['Wire Diameter [m]'][0] # Wire diameter in meters
    a   = k / (rho * c) # Thermal diffusivity in m^2/s
    Q   = np.linspace(df['Min Q [W]'][0], df['Max Q [W]'][0], df['Q step'][0]) # Heat input in watts
    x   = np.arange(-df['x dim'][0], df['x dim'][0]+df['mesh step'][0], df['mesh step'][0] ) 
    y   = np.arange(0, df['y dim'][0]+df['mesh step'][0], df['mesh step'][0] ) 
    z   = np.arange(0, df['z dim'][0]+df['mesh step'][0], df['mesh step'][0] ) 
    X, Y, Z = np.meshgrid(x, y, z,indexing = 'ij') # Create a 3D grid of points
    R   = np.sqrt(X**2 + Y**2 + Z**2) 
    b = []
    d = []
    # Calculate the temperature distribution using Rosenthal's equation
    T   = np.zeros((len(x), len(y), len(z))) # Initialize temperature array
    weld_width = np.zeros((len(v), len(Q))) # Initialize weld width array
    weld_length = np.zeros((len(v), len(Q))) # Initialize weld length array
    
    # XY plane at surface (z=0) finding width & length of weld
    for i in range(len(v)):
        for j in range(len(Q)):
            for l in range(len(x)):
                for m in range(len(y)):
                    T[l][m][0] = ((B_T + ((eta * Q[j]) / (2 * np.pi * k * R[l,m,0])) * np.exp(-v[i] * (R[l,m,0]-x[l]) / (2 * a))))
                    #print(l,m)
                    if T[l][m][0] >= L:
                        T[l][m][0] = 0
            ix = np.where(T[:,0,0] ==0)
            weld_length[i][j] = (abs(x[np.max(ix)] + abs(x[np.min(ix)])))*1000 # weld length converted to mm
            for i_3 in range(2,len(ix)):
                iy = np.where(T[ix[i_3],:,0] == 0)
                print(iy)
                b[i_3] = len(iy[i_3])
                if b[i_3] > b[i_3-1]:
                    i_y = iy
                
            weld_width[i][j] = 2 * y[i_y] *1000 # weld width converted to mm 
        
        print(f'percentage complete: {round((i+1)/len(v)*100,2)}%')
    #pd.DataFrame(weld_width).to_excel(csv_filename, sheet_name='Output', startrow=0, header=False) # do not use it will delete all other sheets in the excel file. Consider creating a new csv or excel as an output. I am unfathemably angry at the moment and will be rewriting the beautifully crafted code I have written to save the output to a csv file instead of an excel file. Perhhaps I will go walk it off. 
    #pd.DataFrame(weld_length).to_excel(csv_filename, sheet_name='Output', startrow=len(weld_width)+2, header=False)
            
            

                        

    

    # XZ plane at centerline (Y=max width) finding depth of weld




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
