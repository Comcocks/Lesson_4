import torch

from convolutional_basics.datasets import get_mnist_loaders, get_cifar_loaders
from models.cnn_models import SimpleCNN, CNNWithResidual, CIFARCNN
from models.fc_models import FCModel, FCModelCIFAR
from utils.visualization_utils import plot_confusion_matrix, plot_gradient
from utils.comparison_utils import compare_models, run_model, save_results


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models_mist = {
    "fc": FCModel(input_size=784, num_classes=10, hidden_sizes=[256, 128, 64]).to(device),
    "simple_cnn": SimpleCNN(input_channels=1, num_classes=10).to(device),
    "residual_cnn": CNNWithResidual(input_channels=1, num_classes=10).to(device)
}
models_cifar = {
    "fc_cifar": FCModelCIFAR().to(device),
    "residual_cnn_cifar": CIFARCNN(num_classes=10, dropout=None).to(device),
    "dropout_residual_cnn": CIFARCNN(num_classes=10, dropout=0.4).to(device)
}


def compare_mnist():
    train_loader, test_loader = get_mnist_loaders(batch_size=64)

    results = {}
    for n, m in models_mist.items():
        results[n] = run_model(n, m, train_loader, test_loader, device, "plots/cnn_vs_fc")

    compare_models(results, save_path="plots/cnn_vs_fc/fc_vs_cnn_mist_compare.png")
    save_results(results, "results/mnist_comparison/results.json")


def compare_cifar():
    train_loader, test_loader = get_cifar_loaders(batch_size=64)

    results = {}
    for n, m in models_cifar.items():
        results[n] = run_model(n, m, train_loader, test_loader, device, "plots/cnn_vs_fc")

    compare_models(results, save_path="plots/cnn_vs_fc/fc_vs_cnn_cifar_compare.png")
    save_results(results, "results/cifar_comparison/results.json")

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for n, m in models_cifar.items():
        plot_confusion_matrix(m, test_loader, classes, device, save_path=f"plots/cnn_vs_fc/{n}_cm.png")

    model = models_cifar["dropout_residual_cnn"]
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    model.zero_grad()
    criterion(model(x), y).backward()
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = param.grad.norm().item()
    plot_gradient(norms, f'plots/cnn_vs_fc/gradient_flow.png')


if __name__ == "__main__":
    print(f"Using device: {device}")

    #compare_mnist()
    compare_cifar()