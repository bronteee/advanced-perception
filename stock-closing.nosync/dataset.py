import torch
import pandas as pd
import numpy as np
from typing import Literal

DATA_FILE_DIR = 'data/optiver-trading-at-the-close/train.csv'
DROP_FEATURES = [
    # 'far_price',
    # 'near_price',
    'time_id',
    'row_id',
]
MAX_SECONDS = 55  # Maximum number of seconds * 10 in a window


def load_and_clean_data(
    data_filepath: str, fillna: Literal['zero', 'mean'] = 'mean'
) -> pd.DataFrame:
    # Load data from csv file
    data = pd.read_csv(data_filepath)
    # Drop features
    data = data.drop(columns=DROP_FEATURES)
    if fillna == 'zero':
        # Replace all NaN values with 0
        data = data.fillna(0)
    elif fillna == 'mean':
        # Replace all NaN values in far_price and near_price with column mean
        data = data.fillna(data.mean())
    else:
        raise ValueError(f"fillna must be 'zero' or 'mean', not {fillna}.")
    return data


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
            window[-idx:, :] = self.data[:idx, :]
        else:
            window = self.data[idx - self.window_size : idx, :]
        # Convert window to tensor
        window = torch.from_numpy(window).float()
        # Get target and convert to numpy array
        target = self.targets[idx].reshape(1)
        # Convert target to tensor
        target = torch.from_numpy(target).float()
        return window, target


if __name__ == '__main__':
    # Test dataset
    dataset = StockDataset(DATA_FILE_DIR)
    print(len(dataset))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    for batch_idx, (data, target) in enumerate(train_loader):
        print(data.shape)
        print(target.shape)
        break
