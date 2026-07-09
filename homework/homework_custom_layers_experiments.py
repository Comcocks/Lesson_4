import torch

from models.custom_layers import (
    CustomConv, SpatialAttention, MISH, AdaptivePool,
    BasicResidual, BottleneckResidual, WideResidual,
    BaselineCNN, MishCNN, AttentionCNN, ResidualCNN
)
from convolutional_basics.datasets import get_cifar_loaders
from utils.comparison_utils import compare_models, run_model, save_results


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, test_loader = get_cifar_loaders(batch_size=128)


def test_forward():
    x = torch.randn(2, 3, 64, 64)

    # Custom conv
    ca = CustomConv(3, 64, 3)
    out = ca(x)
    assert out.shape == x.shape, f'ChannelAttention shape mismatch: {out.shape}'

    # SpatialAttention
    sa = SpatialAttention()
    out = sa(x)
    assert out.shape == x.shape

    # Mish
    inp = torch.linspace(-3, 3, 100)
    mish = MISH()
    mi_out = mish(inp)

    # Adaptive Pooling
    pool = AdaptivePool(3)
    out = pool(x)
    assert out.shape == (4, 64, 1, 1)

    # Residual blocks
    for block, name in [(BasicResidual, 'Basic'), (BottleneckResidual, 'Bottleneck'), (WideResidual, 'Wide')]:
        blk = block(64)
        out = blk(x)
        assert out.shape == x.shape


def compare_custom_layers():

    models = {
        'BaselineCNN': BaselineCNN().to(device),
        'MishCNN': MishCNN().to(device),
        'AttentionCNN': AttentionCNN().to(device),
    }

    results = {}
    for n, m in models.items():
        results[n] = run_model(n, m, train_loader, test_loader, device, "plots/custom_layers")

    compare_models(results, save_path="plots/custom_layers/layers_compare.png")
    save_results(results, "results/custom_layers/layers.json")


def compare_residual_blocks():
    models = {
        'ResidualBasic': ResidualCNN(block_type='basic').to(device),
        'ResidualBottleneck': ResidualCNN(block_type='bottleneck').to(device),
        'ResidualWide': ResidualCNN(block_type='wide').to(device),
    }

    results = {}
    for n, m in models.items():
        results[n] = run_model(n, m, train_loader, test_loader, device, "plots/custom_layers")

    compare_models(results, save_path="plots/custom_layers/layers_compare.png")
    save_results(results, "results/custom_layers/residual.json")


if __name__ == '__main__':
    print(f"Using device: {device}\n")

    test_forward()
    compare_custom_layers()
    compare_residual_blocks()