import matplotlib.pyplot as plt
import numpy as np

y1 = np.array([0.25,0.52,0.56,0.60,0.90,0.75,0.15,0.36,0.55,0.30])
y2 = np.array([0.65,0.95,0.80,0.20,0.80,0.65,0.55,0.97,0.85,0.40])

y3 = y2-y1

x = np.array([i for i in range(1,11)], dtype=int)

fig, subs = plt.subplots(3)

subs[0].plot(x,y1,label='X1')
subs[0].legend()
subs[1].plot(x,y2,label='X2')
subs[1].legend()

subs[2].plot(x,y3,label='X\'')
subs[2].legend()

plt.show()