import json
import matplotlib.pyplot as plt
import time

from utils.training_utils import train_model
from utils.visualization_utils import plot_training_history


def count_parameters(model):
    """Подсчитывает количество параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compare_models(results, save_path):
    """Сравнивает результаты полносвязной и сверточной сетей"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for n, m in results.items():
        ax1.plot(m["history"]["test_accs"], label=f"{n}: {m["time"]:.2f} s", marker='o')
        ax2.plot(m["history"]["test_losses"], label=f"{n}: {m["time"]:.2f} s", marker='o')

    ax1.set_title('Test Accuracy Comparison')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title('Test Loss Comparison')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def run_model(name, model, train_loader, test_loader, device, path):
    result = {}

    print(f"Training {name}...")
    start = time.time()
    result["history"] = train_model(model, train_loader, test_loader, epochs=10, device=device)
    result["time"] = time.time() - start
    result["params"] = count_parameters(model)

    plot_training_history(result["history"], save_path=f"{path}/{name}_training.png")

    return result


def save_results(results, path):
    result = {}
    for n, m in results.items():
        result[n] = {"test_acc": m["history"]["test_accs"][-1],
                     "train_acc": m["history"]["train_accs"][-1],
                     "time": m["time"],
                     "params": m["params"]}

    json.dump(result, open(path, 'w'), indent=2, ensure_ascii=False)