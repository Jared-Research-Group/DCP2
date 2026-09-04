import numpy as np
import matplotlib.pyplot as plt

data = np.load(r"D:\MASON\Data\LSTM\test_in\0073_0024.npy", allow_pickle = True)

print(data.shape)

plt.plot(np.arange(len(data)), data)
plt.show()