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


def compare_custom_layers():

    models = {
        'BaselineCNN': BaselineCNN().to(device),
        'MishCNN': MishCNN().to(device),
        'AttentionCNN': AttentionCNN().to(device),
    }

    results = {}
    for n, m in models.items():
        results[n] = run_model(n, m, train_loader, test_loader, device, "plots/custom_layers")

    compare_models(results, save_path="plots/custom_layers/customlayers_compare.png")
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

    compare_custom_layers()
    #compare_residual_blocks()