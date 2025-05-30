import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.MaxPool2d(2)

        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, n_classes, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        x = self.decoder(x)
        return torch.sigmoid(x)


