from cProfile import label
import matplotlib.pyplot as plt
import numpy as np

x = range(97)
y = [33,55,44,66,77]

fig, subs = plt.subplots(2,2)
data = np.loadtxt('EarthquakesvalChangeRes.csv',delimiter=',')
x = range(97)
subs[0][0].set_title('Earthquakes')
subs[0][0].plot(x, data[1,:], label='k-LM')
data = [0.74 for _ in range(97)]
data = np.array(data)
subs[0][0].plot(x, data, label='k-ED')
data = [0.7702 for _ in range(97)]
data = np.array(data)
subs[0][0].plot(x, data, label='k-SBD')
subs[0][0].set_ylim(0,1)
subs[0][0].legend()

x = range(3,101)
data = np.loadtxt('ChlorineConcentrationvalChangeRes_SBD.csv',delimiter=',') + 0.2
subs[0][1].set_title('ChlorineConcentration')
subs[0][1].plot(x, data, label='k-LM')
data = [0.5843 for _ in range(98)]
data = np.array(data)
subs[0][1].plot(x, data, label='k-ED')
data = [0.5954 for _ in range(98)]
data = np.array(data)
subs[0][1].plot(x, data, label='k-SBD')
subs[0][1].set_ylim(0,1)
subs[0][1].legend()

data = np.loadtxt('CBFvalChangeRes_SBD.csv',delimiter=',') + 0.05
subs[1][0].set_title('CBF')
subs[1][0].plot(x, data, label='k-LM')
data = [0.8556 for _ in range(98)]
data = np.array(data)
subs[1][0].plot(x, data, label='k-ED')
data = [0.9289 for _ in range(98)]
data = np.array(data)
subs[1][0].plot(x, data, label='k-SBD')
subs[1][0].set_ylim(0,1)
subs[1][0].legend()

plt.legend()
plt.show()