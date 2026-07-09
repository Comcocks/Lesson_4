import numpy
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


def plot_training_history(history, save_path):
    """Визуализирует историю обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title('Loss')
    ax1.legend()

    ax2.plot(history['train_accs'], label='Train Acc')
    ax2.plot(history['test_accs'], label='Test Acc')
    ax2.set_title('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(model, loader, classes, device, save_path):
    model.eval()

    preds_list, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds = model(x.to(device)).argmax(1).cpu()
            preds_list.extend(preds.tolist())
            labels.extend(y.tolist())

    cm = confusion_matrix(labels, preds_list)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap='Oranges', vmin=0, vmax=1)
    plt.colorbar(im)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', fontsize=7, color='white' if cm_norm[i, j] > 0.5 else 'black')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_gradient(grads, save_path):
    layers = list(grads.keys())

    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.4), 5))
    ax.plot(list(grads.values()), marker='o')

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=90, fontsize=6)
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Flow")
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_feature_maps(model, image, save_path):
    activations = []
    def hook_fn(module, inp, out):
        activations.append(out.detach())

    handle = model.conv1.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        model(image.unsqueeze(0) if image.dim() == 3 else image)
    handle.remove()

    batch, channels, height, width = activations[0].shape

    num_maps = min(32, channels)
    cols = 4
    rows = (num_maps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = numpy.array(axes).ravel()
    plt.suptitle(f'Feature maps: conv1', fontsize=16)

    for i in range(num_maps):
        axes[i].imshow(activations[0][0, i].cpu().numpy(), cmap='viridis')
        axes[i].set_title(f'ch {i}', fontsize=8)
        axes[i].axis('off')
    for i in range(num_maps, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()