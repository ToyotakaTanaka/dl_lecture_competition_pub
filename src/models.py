import torch
import torch.nn as nn

class VGG19_1D(nn.Module):
    def __init__(self, num_classes, input_channels, seq_len):
        super(VGG19_1D, self).__init__()
        self.features = nn.Sequential(
            self._make_layer(input_channels, 64, 2),
            self._make_layer(64, 128, 2),
            self._make_layer(128, 256, 2),
            self._make_layer(256, 512, 2),
            self._make_layer(512, 512, 2, pool=False),
        )
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
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
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# BasicConvClassifierの代わりにこのモデルを使用
def get_model(num_classes, seq_len, in_channels):
    return VGG19_1D(num_classes, in_channels, seq_len)