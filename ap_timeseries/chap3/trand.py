import numpy as np
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import pandas as pd


csv_path = 'C:/home/myproj/study_ml/ap_timeseries/chap1/AirPassengers.csv'


def FGHset(n_dim_trend, n_dim_obs=1, n_dim_series=0, Q_sigma2=10):
    n_dim_Q = (n_dim_trend != 0) + (n_dim_series != 0)
    if n_dim_series > 0:
        n_dim_state = n_dim_trend + n_dim_series - 1
    else:
        n_dim_state = n_dim_trend

    G = np.zeros((n_dim_state, n_dim_Q))
    F = np.zeros((n_dim_state, n_dim_state))
    H = np.zeros((n_dim_obs, n_dim_state))
    Q = np.eye(n_dim_Q) + Q_sigma2

    G[0, 0] = 1
    H[0, 0] = 1

    if n_dim_trend == 1:
        F[0, 0] = 1
    elif n_dim_trend == 2:
        F[0, 0] = 2
        F[0, 1] = -1
        F[1, 0] = 1
    elif n_dim_trend == 3:
        F[0, 0] = 3
        F[0, 1] = -3
        F[0, 2] = 1
        F[1, 0] = 1
        F[2, 1] = 1

    Q = G.dot(Q).dot(G.T)

    return n_dim_state, F, H, Q

df_content = pd.read_csv(csv_path)
df_content['Month'] = pd.to_datetime(df_content['Month'], infer_datetime_format=True)
y = pd.Series(df_content["#Passengers"].values, index=df_content['Month'])
y = y.astype('f')

n_dim_obs = 1
n_dim_trend = 2

n_dim_state, F, H, Q = FGHset(n_dim_trend, n_dim_obs)

initial_state_mean = np.zeros(n_dim_state)
initial_state_covariance = np.ones((n_dim_state, n_dim_state))

kf = KalmanFilter(
    n_dim_obs=n_dim_obs,
    n_dim_state=n_dim_state,
    initial_state_mean=initial_state_mean,
    initial_state_covariance=initial_state_covariance,
    transition_matrices=F,
    observation_matrices=H,
    transition_covariance=Q
)

n_train = 120
train, test = y.values[:n_train], y.values[n_train:]

filtered_state_means, filtered_state_covs = kf.filter(train)
smoothed_state_means, smoothed_state_covs = kf.smooth(train)

pred_o_smoothed = smoothed_state_means.dot(H.T)

#plt.plot(train, label="observation")
#plt.plot(pred_o_smoothed, '--', label="predict")
#plt.show()

plt.plot(y.values, label="observation")

pred_y = np.empty(len(test))

current_state = smoothed_state_means[-1]
current_cov = smoothed_state_covs[-1]

for i in range(len(test)):
    current_state, current_cov = kf.filter_update(
        current_state, current_cov, observation=None
    )

    pred_y[i] = kf.observation_matrices.dot(current_state)

plt.plot(np.hstack([pred_o_smoothed.flatten(), pred_y]), '--', label="forecast")
plt.show()

