import sys
from data_manipulation import selectFolder
from align_data import alignData
from batch_process import dataSearch

def batchit(dir):
    dir = dir.path
    alignData(dir, True)
    return

def main():
    if len(sys.argv) != 2:
        dir = selectFolder()
    else:
        dir = sys.argv[1]
    
    #alignData(dir, True)
    dataSearch(dir, batchit)
    return  

if __name__ == '__main__': main()