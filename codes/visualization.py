"""Generate detailed visualizations for the project report.

Outputs (saved under codes/figs/):
- confusion_matrix_mlp.png
- confusion_matrix_cnn.png
- misclassified_mlp.png
- misclassified_cnn.png
- mlp_weights.png
- cnn_kernels.png
"""
import gzip
from pathlib import Path
from struct import unpack

import matplotlib.pyplot as plt
import mynn as nn
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / 'figs'
FIG_DIR.mkdir(exist_ok=True)
MLP_CKPT = BASE_DIR / 'saved_models' / 'mlp_baseline' / 'best_model.pickle'
CNN_CKPT = BASE_DIR / 'saved_models' / 'cnn_reg_mild' / 'best_model.pickle'


def load_test_set():
    test_images_path = BASE_DIR / 'dataset' / 'MNIST' / 't10k-images-idx3-ubyte.gz'
    test_labels_path = BASE_DIR / 'dataset' / 'MNIST' / 't10k-labels-idx1-ubyte.gz'
    with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)
    with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        labs = np.frombuffer(f.read(), dtype=np.uint8)
    imgs = imgs.astype(np.float32) / 255.0
    return imgs[:1000], labs[:1000]


def predict(model, X):
    logits = model(X)
    return np.argmax(logits, axis=-1)


def plot_confusion_matrix(true_labels, pred_labels, save_path, title):
    num_classes = 10
    matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(true_labels, pred_labels):
        matrix[t, p] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap='Blues')
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                    color='white' if matrix[i, j] > matrix.max() / 2 else 'black',
                    fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_misclassified(images_flat, true_labels, pred_labels, save_path, title, max_count=12):
    wrong_idx = np.where(true_labels != pred_labels)[0]
    if len(wrong_idx) == 0:
        return
    selection = wrong_idx[:max_count]
    cols = 6
    rows = int(np.ceil(len(selection) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.8))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis('off')
    for ax, idx in zip(axes, selection):
        img = images_flat[idx].reshape(28, 28)
        ax.imshow(img, cmap='gray')
        ax.set_title(f'T:{true_labels[idx]} P:{pred_labels[idx]}', fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_mlp_weights(model, save_path):
    first_linear = None
    for layer in model.layers:
        if isinstance(layer, nn.op.Linear):
            first_linear = layer
            break
    if first_linear is None:
        return
    W = first_linear.params['W']  # [784, hidden]
    num = min(16, W.shape[1])
    cols = 4
    rows = int(np.ceil(num / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis('off')
    for i in range(num):
        ax = axes[i]
        ax.imshow(W[:, i].reshape(28, 28), cmap='seismic')
        ax.set_title(f'h{i}', fontsize=8)
    fig.suptitle('MLP first-layer weights')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_cnn_kernels(model, save_path):
    first_conv = None
    for layer in model.layers:
        if isinstance(layer, nn.op.conv2D):
            first_conv = layer
            break
    if first_conv is None:
        return
    W = first_conv.params['W']  # [out, in, k, k]
    num = W.shape[0]
    cols = 4
    rows = int(np.ceil(num / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis('off')
    for i in range(num):
        ax = axes[i]
        ax.imshow(W[i, 0], cmap='seismic')
        ax.set_title(f'k{i}', fontsize=8)
    fig.suptitle('CNN first-layer 3x3 kernels')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def main():
    test_flat, test_labs = load_test_set()
    test_cnn = test_flat.reshape(-1, 1, 28, 28)

    mlp = nn.models.Model_MLP()
    mlp.load_model(MLP_CKPT)
    mlp_pred = predict(mlp, test_flat)
    plot_confusion_matrix(test_labs, mlp_pred, FIG_DIR / 'confusion_matrix_mlp.png', 'MLP confusion matrix')
    plot_misclassified(test_flat, test_labs, mlp_pred, FIG_DIR / 'misclassified_mlp.png', 'MLP misclassified examples')
    plot_mlp_weights(mlp, FIG_DIR / 'mlp_weights.png')

    cnn = nn.models.Model_CNN()
    cnn.load_model(CNN_CKPT)
    cnn_pred = predict(cnn, test_cnn)
    plot_confusion_matrix(test_labs, cnn_pred, FIG_DIR / 'confusion_matrix_cnn.png', 'CNN confusion matrix')
    plot_misclassified(test_flat, test_labs, cnn_pred, FIG_DIR / 'misclassified_cnn.png', 'CNN misclassified examples')
    plot_cnn_kernels(cnn, FIG_DIR / 'cnn_kernels.png')

    print('Saved figures to', FIG_DIR)


if __name__ == '__main__':
    main()
