import numpy as np
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import pandas as pd
import io
from hmmlearn.hmm import GaussianHMM

#table_file = "d:/home/myproj/study_ml/ap_timeseries/chap3/ew_excs.prn.txt"
table_file = "D:/myproj/study_ml/ap_timeseries/chap3/ew_excs.prn.txt"
raw = pd.read_table(table_file, header=None, engine='python', skipfooter=1)
raw.index = pd.date_range('1926-01-01', '1995-12-01', freq='MS')
quotes = raw.loc[:'1986'] - raw.loc[:'1986'].mean()


#quotes[0].plot(figsize=(12,3))

model = GaussianHMM(n_components=3, covariance_type="full", n_iter=5000).fit(quotes)
hidden_state = model.predict(quotes)

colors = ['b', 'Y', 'r']

for i, color, in enumerate(colors):
    mask = hidden_state != i
    tmp = quotes.copy()
    tmp[mask] = None
    plt.plot(tmp, ".", c=color)

plt.xlim('1926-01-01', '1995-12-01')
plt.grid(True)

plt.show()

