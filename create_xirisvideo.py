import matplotlib.pyplot as plt
import numpy as np
import math
from tkinter import filedialog

def convertFrame(f):
    
    frame = open(f, "rb")
    frame.read(16)
    
    width = int.from_bytes(frame.read(4), 'little')
    height = int.from_bytes(frame.read(4), 'little')
    
    frame.read(32)
    
    #frame.read(height*width*2)
    
    min = math.pow(2,16)
    max = 0
    intensity = np.zeros((height, width))
    for x in range(height):
        for y in range(width):
            intensity[x][y] = int.from_bytes(frame.read(2), 'little')
            if intensity[x][y] < min: min = intensity[x][y]
            if intensity[x][y] > max: max = intensity[x][y]
    
    '''
    scale = max - min
    for x in range(height):
        for y in range(width):
            intensity[x][y] = int(((intensity[x][y] - min) / scale) * (math.pow(2,16) - 1))
            '''
            
            
    # what is the 2nd half of the .raw file? almost solidly colored image. we have 2x arrays of image's shape. currently only using the 1st one
    # operating on assumption that 2nd half is a 'dark frame'. intended to allow us to normalize to a blank capture from the sensor. subtract dark frame from captured frame for true measurement
    
    dark_frame = np.zeros((height, width))
    for x in range(height):
        for y in range(width):
            dark_frame[x][y] = int.from_bytes(frame.read(2), 'little')
            
            intensity[x][y] -= dark_frame[x][y]
            
    
    
    return intensity

def main():
    frame = filedialog.askopenfilename()
    
    framePNG = convertFrame(frame)
    plt.imshow(framePNG, cmap='twilight')
    plt.show()
    return  

if __name__ == '__main__': main()