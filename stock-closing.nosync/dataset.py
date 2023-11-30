import torch
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Literal

DATA_FILE_DIR = './data/train_added_features.csv'
TRAIN_TARGET_SERIES_DATA_FILE_DIR = './data/train_target_series.csv'
VALIDATION_TARGET_SERIES_DATA_FILE_DIR = './data/valid_target_series.csv'
MAX_SECONDS = 55  # Maximum number of seconds * 10 in a window


def load_and_clean_data(
    data_filepath: str,
    fillna: Literal['zero', 'mean'] = 'mean',
    add_features_flag: bool = True,
) -> pd.DataFrame:
    """
    Load and clean data from csv file.
    Args:
        data_filepath (string): Path to the csv file with stock data.
        fillna (string): How to fill NaN values. Default: 'mean'.
    Returns:
        data (DataFrame): Cleaned data.
    """
    # Load data from csv file
    data = pd.read_csv(data_filepath)
    if fillna == 'zero':
        # Replace all NaN values with 0
        data = data.fillna(0)
    elif fillna == 'mean':
        # Replace all NaN values in far_price and near_price with column mean
        data = data.fillna(data.mean())
    else:
        raise ValueError(f"fillna must be 'zero' or 'mean', not {fillna}.")
    if add_features_flag:
        data = sizesum_and_pricestd(data)

    return data


def sizesum_and_pricestd(df) -> pd.DataFrame:
    price_ftrs = [
        'reference_price',
        'far_price',
        'near_price',
        'bid_price',
        'ask_price',
        'wap',
    ]  # std
    size_ftrs = ['imbalance_size', 'matched_size', 'bid_size', 'ask_size']  # sum

    rolled = (
        df[['stock_id'] + size_ftrs]
        .groupby('stock_id')
        .rolling(window=6, min_periods=1)
        .sum()
    )
    rolled = rolled.reset_index(level=0, drop=True)
    for col in size_ftrs:
        df[f'{col}_rolled_sum'] = rolled[col]

    rolled = (
        df[['stock_id'] + price_ftrs]
        .groupby('stock_id')
        .rolling(window=6, min_periods=1)
        .std()
        .fillna(0)
    )
    rolled = rolled.reset_index(level=0, drop=True)
    for col in price_ftrs:
        df[f'{col}_rolled_std'] = rolled[col]

    return df


class StockDataset(torch.utils.data.Dataset):
    """
    Define a dataset for Optiver stock movement prediction.
    """

    def __init__(self, data_filepath: str, window_size: int = 10) -> None:
        """
        Args:
            data_filepath (string): Path to the csv file with stock data.
            window_size (int): Size of the window in 10 seconds for the stock data. Default: 10.
        """
        data = load_and_clean_data(data_filepath)
        # data = data.drop(columns=DROP_FEATURES) not needed, already removed in added features training csv
        self.data = data.drop(columns=["target"]).to_numpy()
        self.targets = data["target"].to_numpy()
        assert window_size > 0, "Window size must be greater than 0."
        assert (
            window_size <= MAX_SECONDS
        ), f"Window size must be less than or equal to {MAX_SECONDS}."
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        if idx <= self.window_size:
            # If the index is less than the window size, pad with zeros
            window = np.zeros((self.window_size, self.data.shape[1]))
            if idx == 0:
                window[-1, :] = self.data[0, :]
            else:
                window[-idx:, :] = self.data[:idx, :]
        else:
            window = self.data[idx - self.window_size : idx, :]
        # Convert window to tensor
        window = torch.from_numpy(window).float()
        # Expand window dimensions to (1, window_size, num_features)
        window = torch.unsqueeze(window, 0)
        # Get target and convert to numpy array
        target = self.targets[idx].reshape(1)
        # Convert target to tensor
        target = torch.from_numpy(target).float()
        return window, target


def windowed_dataset(series, window_size=55, batch_size=32):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    return ds.batch(batch_size).prefetch(1)


class TargetTimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data_file_path, window_size=55):
        series = pd.read_csv(data_file_path).to_numpy()
        self.series = torch.tensor(series, dtype=torch.float32)
        print(self.series.shape)
        # make channel the first dimension
        # self.series = self.series.permute(2, 0, 1)
        self.window_size = window_size

    def __len__(self):
        return len(self.series) - self.window_size

    def __getitem__(self, index):
        return (
            self.series[index : index + self.window_size][:,:-1].unsqueeze(0),
            self.series[index + self.window_size][-1].unsqueeze(0),
        )


if __name__ == '__main__':
    # # Test dataset
    # dataset = StockDataset(DATA_FILE_DIR)
    # print(len(dataset))
    # train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    # for batch_idx, (data, target) in enumerate(train_loader):
    #     print(data.shape)
    #     print(target.shape)
    #     print(target)
    #     print(batch_idx)
    #     break

    # Test windowed dataset
    dataset = TargetTimeSeriesDataset(TRAIN_TARGET_SERIES_DATA_FILE_DIR)
    print(len(dataset))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    for batch_idx, (data, target) in enumerate(train_loader):
        print(data.shape)
        print(target.shape)
        print(target)
        print(batch_idx)
        # Make sure the target is not always in the input data
        count = 0
        for i, sample in enumerate(data):
            if target[i] in sample:
                print(f'{target[i]} is in {sample}')
                print(f'Index: {i}')
                count += 1
        print(f'Count: {count}')
        break
