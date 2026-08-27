import sys
import subprocess
import os
from pathlib import Path

from data_manipulation import selectFolder
from align_data import alignData
from batch_process import dataSearch

def batchit(dir):
    dir = Path(dir)

    #subprocess.run([sys.executable, os.getcwd() + '\\Data_Processing.py', dir], check=True)

    print()
    
    alignData(dir, True)
    
    return

def main():
    if len(sys.argv) != 2:
        dir = selectFolder()
    else:
        dir = sys.argv[1]
    
    dataSearch(dir, batchit)
    return  

if __name__ == '__main__': main()