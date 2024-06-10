import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm


class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)

    def forward(self, x):
        x = self.bottleneck(x)
        x = self.bn(x)
        return x


class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256):
        super(feat_classifier, self).__init__()
        self.fc = nn.Linear(bottleneck_dim, class_num)

    def forward(self, x):
        x = self.fc(x)
        return x
