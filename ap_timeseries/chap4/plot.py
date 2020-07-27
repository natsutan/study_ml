import matplotlib.pyplot as plt
import numpy as np

data = np.load("score.npz.npy")[0:800]
data = np.load("score.npz.npy")[0:800]
predict = np.load("predict.npz.npy") / 2.0

plt.plot(data, color='red', marker='+')
plt.plot(predict, color='green', linestyle="dashed")
plt.xlim(0, 800)

plt.show()





