import torch
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm


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
