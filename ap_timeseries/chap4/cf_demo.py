import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import changefinder

data = np.concatenate(
    [np.random.normal(0.7, 0.05, 300),
     np.random.normal(1.5, 0.05, 300),
     np.random.normal(0.6, 0.05, 300),
     np.random.normal(1.3, 0.05, 300)
    ]
)

cf = changefinder.ChangeFinder(r=0.01, order=1, smooth=7)

result = np.empty(len(data))
for i, d in enumerate(data):
    result[i] = cf.update(d)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(result, label="score")
ax2 = ax.twinx()
ax2.plot(data, alpha=0.3, label="observaton")

#plt.plot(data)
plt.show()
