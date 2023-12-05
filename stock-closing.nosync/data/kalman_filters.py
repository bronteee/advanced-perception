from pykalman import KalmanFilter
import pandas as pd

def apply_kalman_filter(data):
    # Initialize the Kalman Filter
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)

    # Use the observed values of the price to get a rolling mean
    state_means, _ = kf.filter(data.values)
    state_means = pd.Series(state_means.flatten(), index=data.index)

    return state_means

# Load the data
data = pd.read_csv('/content/drive/MyDrive/train_windows.csv')

# Apply the Kalman filter to each column
for column in data.columns:
    data[column] = apply_kalman_filter(data[column])

# Save the filtered data
data.to_csv('/content/drive/MyDrive/train_windows_kalman.csv', index=False)
