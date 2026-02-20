import sys
from data_manipulation import selectFolder
from align_data import alignData

def main():
    if len(sys.argv) != 2:
        dir = selectFolder()
    else:
        dir = sys.argv[1]
    
    alignData(dir, True)
    return  

if __name__ == '__main__': main()