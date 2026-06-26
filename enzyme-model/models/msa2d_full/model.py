import torch
import torch.nn as nn


class ContactMapCNN(nn.Module):

    def __init__(self, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1)
        )

        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self._init_weights()

    def _init_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):

                nn.init.kaiming_normal_(
                    m.weight,
                    nonlinearity="relu"
                )

                nn.init.zeros_(m.bias)

    def encode(self, x):

        x = self.features(x)

        x = x.flatten(1)

        return x

    def forward(self, x):

        feat = self.encode(x)

        pred = self.regressor(feat)

        return pred

    @property
    def cnn_feature_dim(self):

        return 128

