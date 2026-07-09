import torch

from utils.comparison_utils import run_model, save_results
from convolutional_basics.datasets import get_cifar_loaders
from models.cnn_models import CNNKernel, CNNKernelMixed, CNNDepth, CIFARCNN
from utils.visualization_utils import plot_feature_maps, plot_gradient
from utils.comparison_utils import compare_models


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, test_loader = get_cifar_loaders(batch_size=128)


def compare_core_size():
    models = {
        'CNN_3': CNNKernel(size=3).to(device),
        'CNN_5': CNNKernel(size=5).to(device),
        'CNN_7': CNNKernel(size=7).to(device),
        'CNN_Mixed_1+3': CNNKernelMixed().to(device),
    }

    results = {}
    for n, m in models.items():
        results[n] = run_model(n, m, train_loader, test_loader, device, "plots/cnn_architecture")

    save_results(results, "results/architecture_analysis/core_size.json")
    compare_models(results, save_path="plots/cnn_architecture/cnn_core_compare.png")

    img = next(iter(test_loader))[0][0]
    for n, m in models.items():
        plot_feature_maps(m.to(device), img.to(device), f'plots/cnn_architecture/{n}_feature_maps.png', )


def compare_depth():
    models = {
        'CNN_shallow': CNNDepth(n_layers=2).to(device),
        'CNN_medium': CNNDepth(n_layers=4).to(device),
        'CNN_deep': CNNDepth(n_layers=6).to(device),
        'CNN_residual': CIFARCNN(dropout=0.4).to(device),
    }

    results = {}
    for n, m in models.items():
        results[n] = run_model(n, m, train_loader, test_loader, device, "plots/cnn_architecture")

    save_results(results, "results/architecture_analysis/depth.json")
    compare_models(results, save_path="plots/cnn_architecture/cnn_depth_compare.png")

    img = next(iter(test_loader))[0][0]
    criterion = torch.nn.CrossEntropyLoss()
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    for n, m in models.items():
        m = m.to(device)
        m.train()
        m.zero_grad()
        criterion(m(x), y).backward()

        norms = {}
        for n, param in m.named_parameters():
            if param.grad is not None:
                norms[n] = param.grad.norm().item()

        plot_gradient(norms, f'plots/cnn_architecture/{n}_gradient.png')
        plot_feature_maps(m.to(device), img.to(device), f'plots/cnn_architecture/{n}_feature_maps.png', )


if __name__ == "__main__":
    #compare_core_size()
    compare_depth()