import torch
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm
from models import ResCNN, StockS4, LSTMRegressor

model_mapping = {
    'rescnn': ResCNN,
    's4': StockS4,
    'LSTM': LSTMRegressor
}

TEST_DATA_DIR = Path('./data/train.csv')

@torch.inference_mode()
def evaluate(
    net,
    dataloader,
    device,
    amp,
    batch_size,
    criterion,
    n_val,
    save_predictions=False,
):
    save_dir = Path(f'predictions_{net.__class__.__name__}')
    os.makedirs(save_dir, exist_ok=True)
    net.eval()
    net.to(device=device)
    torch.set_grad_enabled(False)
    num_val_batches = n_val // batch_size
    val_loss = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for i, batch in tqdm(
            enumerate(dataloader),
            total=num_val_batches,
            desc='Validation round',
            unit='batch',
            leave=False,
        ):
            features, target = batch
            features = features.to(device=device)
            target = target.to(device=device)
            pred = net(features)

            # compute the loss
            val_loss += criterion(
                pred,
                target,
            )
            # TODO
            # if save_predictions:
            # Save predictions as np

    net.train()
    return val_loss / max(num_val_batches, 1)

def main():
    checkpoint_file = 'checkpoint_epoch9_nan.pth'
    model_tyoe = 'LSTM'
    criterion = torch.nn.L1Loss()
    # Instantiate the model
    checkpoint_dir = './checkpoints'
    model = model_mapping[model_tyoe]()
    # Load the model checkpoint
    model_state_dict = torch.load(f'{checkpoint_dir}/{checkpoint_file}')[
        'model_state_dict'
    ]
    model.load_state_dict(model_state_dict)
    # Load the test data
    test_set = StockDataset(TEST_DATA_DIR)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)
    # Evaluate the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_loss = evaluate(
        model, test_loader, device, False, 128, criterion, len(test_set), False
    )
    print(f'Validation loss: {val_loss}')


if __name__ == '__main__':
    main()
