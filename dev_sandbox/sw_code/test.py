import matplotlib.pyplot as plt
import numpy as np

# create some sample data
x = np.arange(0, 10, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

# create the first figure and plot the first set of data
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(x, y1)
ax1.set_title('Plot 1')

# create the second figure and plot the second set of data
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(x, y2)
ax2.set_title('Plot 2')

# display the plots
plt.show()
