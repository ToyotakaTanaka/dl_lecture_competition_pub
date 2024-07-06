import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG19_1D(nn.Module):
    def __init__(self, num_classes, input_channels, seq_len):
        super(VGG19_1D, self).__init__()
        self.features = nn.Sequential(
            self._make_layer(input_channels, 64, 2),
            self._make_layer(64, 128, 2),
            self._make_layer(128, 256, 4),
            self._make_layer(256, 512, 4),
            self._make_layer(512, 512, 4, pool=False),  # 最後のプーリングを削除
        )
        
        # AdaptiveMaxPool1dを使用
        self.avgpool = nn.AdaptiveMaxPool1d(7)
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def _make_layer(self, in_channels, out_channels, num_convs, pool=True):
        layers = []
        for _ in range(num_convs):
            conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
            layers += [conv, nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True)]
            in_channels = out_channels
        if pool:
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X ( b, c, t ): Input tensor
        Returns:
            X ( b, num_classes ): Output tensor
        """
        X = self.blocks(X)
        return self.head(X)

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        return self.dropout(X)

def get_model(num_classes, seq_len, in_channels):
    return VGG19_1D(num_classes, in_channels, seq_len)