from derivative import dxdt     # from pysindy
import numpy as np

import matplotlib.pyplot as plt

def smartFD(x, t):
    return dxdt(x, t, kind='finite_difference', k=100)

t = np.linspace(0, 10, 1000)
x = t*3 + np.random.normal(0, 0.1, 1000)

true = x/t

dx = smartFD(x, t)
dx_trend = dxdt(x, t, kind='trend_filtered', order=0, alpha=1e-2)

print(len(t))
print(len(dx_trend))

plt.plot(t, x, label=r'$f(t)$')
plt.plot(t, dx, label=r'$\frac{df}{dt}(t)$ FD')
plt.plot(t, dx_trend, label=r'$\frac{df}{dt}(t)$ TD')
plt.legend()
plt.show()
