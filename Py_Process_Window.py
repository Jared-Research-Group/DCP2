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
    
    # Calculate the temperature distribution using Rosenthal's equation
    T   = np.zeros((len(x), len(y), len(z))) # Initialize temperature array
    weld_width = np.zeros((len(v), len(Q))) # Initialize weld width array
    weld_length = np.zeros((len(v), len(Q))) # Initialize weld length array
    print(np.shape(weld_width))
    
    # XY plane at surface (z=0)
    for i in range(len(v)):
        for j in range(len(Q)):
            for l in range(len(x)):
                for m in range(len(y)):
                    T[l][m][0] = ((B_T + ((eta * Q[j]) / (2 * np.pi * k * R[l,m,0])) * np.exp(-v[i] * (R[l,m,0]-x[l]) / (2 * a))))
                    #print(l,m)
                    if T[l][m][0] >= L:
                        T[l][m][0] = 0
        ix = np.where(T[:,0,0] ==0)
        weld_length[i][j] = (abs(x[max(ix)] + abs(x[min(ix)])))
            #print(weld_length)
            #weld_width[i][j] = 2 * np.max(y[T[:,:,0] == 0]) 
            #print(weld_width[i][j])
    print(weld_length)
            
            
   
      
    



                             
      
                
            
                

                        

    

    # YZ plane at centerline (x=0)




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
