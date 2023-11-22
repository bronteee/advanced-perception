import torch
import torch.nn as nn
import torch
from torch.autograd import grad
from dataset import StockDataset, DATA_FILE_DIR
from s4.s4d_torch import S4D

class ResCNN(nn.Module):
    """
    Defines a residual CNN architecture for stock prediction.
    This is mostly based on https://github.com/hardyqr/CNN-for-Stock-Market-Prediction-PyTorch/blob/master/source/cnn.py
    """

    def __init__(self):
        super(ResCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.LeakyReLU(0.3)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1), nn.LeakyReLU(0.3)
        )
        self.layer3 = nn.Sequential(nn.Conv2d(8, 32, kernel_size=1), nn.LeakyReLU(0.3))
        self.layer4 = nn.Sequential(nn.Conv2d(32, 2, kernel_size=1), nn.LeakyReLU(0.3))
        self.pl = nn.AvgPool2d((5, 2))
        self.fc = nn.Linear(28, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # print(x.size())
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer35(out)
        out = self.layer4(out)
        # print(out.size())
        out = self.pl(out)
        # print(out.size())
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        out = self.tanh(out)  # TODO: Do we need this?
        return out


class StockS4(nn.Module):
    def __init__(
        self,
        d_input=1, # Number of channels, 1 for this dataset
        d_output=1, # Number of outputs, 1 for this dataset
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
        lr=0.001,
    ):
        super().__init__()

        self.prenorm = prenorm
        self.lr = lr

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, self.lr))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (Batch, Length, d_input)
        """
        x = x.view(
            x.shape[0],
            -1,
            x.shape[1],
        )
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x


if __name__ == '__main__':
    # Test the model
    model = ResCNN()
    # model = StockS4()
    print(model)
    # Test dataset
    dataset = StockDataset(DATA_FILE_DIR)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    for batch_idx, (data, target) in enumerate(train_loader):
        print(data.shape)
        print(target.shape)
        # combine last 2 dimensions
        # data = data.view(
        #     data.shape[0],
        #     -1,
        #     data.shape[1],
        # )
        # print(data.shape)
        output = model(data)
        print(output.shape)
        break
