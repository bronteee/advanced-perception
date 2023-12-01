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

    def __init__(self, target_series=True):
        super(ResCNN, self).__init__()
        if target_series is True:
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.LeakyReLU(0.3)
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 8, kernel_size=3, padding=1), nn.LeakyReLU(0.3)
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(8, 32, kernel_size=1), nn.LeakyReLU(0.3)
            )
            self.layer4 = nn.Sequential(
                nn.Conv2d(32, 2, kernel_size=1), nn.LeakyReLU(0.3)
            )
            self.pl = nn.AvgPool2d((5, 2))
            self.fc1 = nn.Linear(2200, 1100)
            self.fc2 = nn.Linear(1100, 200)
            self.tanh = nn.Tanh()
        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.LeakyReLU(0.3)
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 8, kernel_size=3, padding=1), nn.LeakyReLU(0.3)
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(8, 32, kernel_size=1), nn.LeakyReLU(0.3)
            )
            self.layer4 = nn.Sequential(
                nn.Conv2d(32, 2, kernel_size=1), nn.LeakyReLU(0.3)
            )
            self.pl = nn.AvgPool2d((5, 2))
            self.fc1 = nn.Linear(1364, 682)
            self.fc2 = nn.Linear(682, 1)
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
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.tanh(out)  # TODO: Do we need this?
        return out

# Maybe we use less stocks
class StockS4(nn.Module):
    def __init__(
        self,
        d_input=1,  # Number of channels, 1 for this dataset
        d_output=200,  # Number of outputs, 1 for this dataset
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
        # print(x.shape)
        x = x.view(
            x.shape[0],
            -1,
            1,  # Reshape to (Batch, Length, d_input=1) for our data set
        )
        # print(x.shape)
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


class LSTMRegressor(nn.Module):
    def __init__(
        self, input_size=124, hidden_size=64, num_layers=2, output_size=1, dropout=0.2
    ):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Reshape input to (batch_size * sequence_length, input_size)
        x = x.view(x.size(0), -1, x.size(-1))

        # LSTM layer
        lstm_out, _ = self.lstm(x)

        # Take the output from the last time step
        last_hidden_state = lstm_out[:, -1, :]

        # Fully connected layer
        output = self.fc(last_hidden_state)
        return output


class SimpleTransformer(nn.Module):
    def __init__(self, feature_num=124, d_model=96, nhead=8, num_layers=1):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Linear(feature_num, d_model)
        self.tf1 = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.5)
        self.tf2 = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            batch_first=True,
        )
        self.decoder = nn.Linear(d_model, 200)

    def forward(self, x):
        # Reduce the feature dimension
        x = x.view(x.size(0), -1, x.size(-1))
        x = self.embedding(x)
        x = self.tf1.encoder(x)
        x = self.fc(x[:, -1, :])  # Use the last sequence output
        x = self.dropout(x)
        x = self.tf2.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    # Test the model
    # model = ResCNN(target_series=True)
    # model = StockS4()
    # model = LSTMRegressor(input_size=200, output_size=200)
    model = SimpleTransformer(feature_num=200)

    # Test on random input
    x = torch.randn(64, 1, 55, 200)
    output = model(x)
    print(output.shape)
