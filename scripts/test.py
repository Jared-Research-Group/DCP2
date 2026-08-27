from align_data import alignData
from data_manipulation import selectFolder
import pandas as pd

dir = selectFolder()

print(alignData(dir, True))