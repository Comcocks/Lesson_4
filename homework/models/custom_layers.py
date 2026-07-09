import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomConv(nn.Module):
    """
    Кастомный сверточный слой с:
    - L2-нормализацией весов
    - Dropout для сверточных фильтров
    """

    def __init__(self, in_ch, out_ch, kernel_size, bias=True, dropout_rate=0.1, weight_norm=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size//2, bias=bias)
        self.dropout_rate = dropout_rate
        self.weight_norm = weight_norm

        # Инициализация весов
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        weight = self.conv.weight

        # L2-нормализация весов
        if self.weight_norm:
            weight = F.normalize(weight, p=2, dim=(1, 2, 3))

        # Свёртка с нормализованными весами
        x = F.conv2d(x, weight, bias=self.conv.bias, padding=self.conv.padding)

        # Dropout для каналов
        if self.training and self.dropout_rate > 0:
            batch_size, channels, h, w = x.size()
            mask = torch.bernoulli(torch.ones(1, channels, 1, 1) * (1 - self.dropout_rate))
            mask = mask.to(x.device)
            x = x * mask / (1 - self.dropout_rate)

        return x


class SpatialAttention(nn.Module):
    """Пространственный attention механизм"""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.conv(concat))
        return x * attention


class MISH(nn.Module):
    """Кастомная функция активации Mish"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class AdaptivePool(nn.Module):
    """Адаптивный смешанный pooling"""

    def __init__(self, kernel_size, stride=None, padding=0, alpha=0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.alpha = alpha  # вес для max pooling

    def forward(self, x):
        max_pool = F.max_pool2d(x, self.kernel_size, self.stride, self.padding)
        avg_pool = F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)

        return self.alpha * max_pool + (1 - self.alpha) * avg_pool


class BasicResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class BottleneckResidual(nn.Module):
    def __init__(self, channels, ratio=4):
        super().__init__()
        mid = max(channels // ratio, 1)
        self.conv1 = nn.Conv2d(channels, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(out + x)


class WideResidual(nn.Module):
    def __init__(self, channels, width_mult=2):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels * width_mult, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels * width_mult)
        self.conv2 = nn.Conv2d(channels * width_mult, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class ResidualCNN(nn.Module):
    """CNN с выбираемым Residual"""

    def __init__(self, block_type='basic', num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        blocks = {'basic': BasicResidual, 'bottleneck': BottleneckResidual, 'wide': WideResidual}
        block = blocks[block_type]
        self.layer1 = block(64)
        self.pool = nn.MaxPool2d(2)
        self.layer2 = block(64)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.pool(self.layer1(x))
        x = self.layer2(x)
        return self.classifier(x)


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, num_classes))
    def forward(self, x):
        return self.net(x)


class MishCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), MISH(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), MISH(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, num_classes))
    def forward(self, x):
        return self.net(x)


class AttentionCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.sa = SpatialAttention()
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(AdaptivePool(3, alpha=0.6), nn.Flatten(), nn.Linear(512, 128))

    def forward(self, x):
        x = self.stem(x)
        x = self.sa(x)
        x = self.conv2(x)
        return self.classifier(x)