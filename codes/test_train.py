import gzip
import pickle
from pathlib import Path
from struct import unpack

import matplotlib.pyplot as plt
import mynn as nn
import numpy as np
from draw_tools.plot import plot

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'dataset' / 'MNIST'
RESULTS_DIR = BASE_DIR / 'saved_models'
SEED = 309

np.random.seed(SEED)


def load_mnist(train_limit=4000, valid_limit=1000, test_limit=1000):
    train_images_path = DATA_DIR / 'train-images-idx3-ubyte.gz'
    train_labels_path = DATA_DIR / 'train-labels-idx1-ubyte.gz'
    test_images_path = DATA_DIR / 't10k-images-idx3-ubyte.gz'
    test_labels_path = DATA_DIR / 't10k-labels-idx1-ubyte.gz'

    with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

    with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)

    with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

    with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

    idx = np.random.permutation(np.arange(train_labs.shape[0]))
    with open(BASE_DIR / 'idx.pickle', 'wb') as f:
        pickle.dump(idx, f)

    train_imgs = train_imgs[idx]
    train_labs = train_labs[idx]

    valid_imgs = train_imgs[:10000]
    valid_labs = train_labs[:10000]
    train_imgs = train_imgs[10000:]
    train_labs = train_labs[10000:]

    if train_limit is not None:
        train_imgs = train_imgs[:train_limit]
        train_labs = train_labs[:train_limit]
    if valid_limit is not None:
        valid_imgs = valid_imgs[:valid_limit]
        valid_labs = valid_labs[:valid_limit]
    if test_limit is not None:
        test_imgs = test_imgs[:test_limit]
        test_labs = test_labs[:test_limit]

    train_imgs = train_imgs.astype(np.float32) / 255.0
    valid_imgs = valid_imgs.astype(np.float32) / 255.0
    test_imgs = test_imgs.astype(np.float32) / 255.0

    return {
        'train_flat': train_imgs,
        'valid_flat': valid_imgs,
        'test_flat': test_imgs,
        'train_cnn': train_imgs.reshape(-1, 1, 28, 28),
        'valid_cnn': valid_imgs.reshape(-1, 1, 28, 28),
        'test_cnn': test_imgs.reshape(-1, 1, 28, 28),
        'train_labels': train_labs,
        'valid_labels': valid_labs,
        'test_labels': test_labs,
    }


def evaluate_model(model, X, y):
    logits = model(X)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=int(np.max(y)) + 1)
    loss = loss_fn(logits, y)
    acc = nn.metric.accuracy(logits, y)
    return float(loss), float(acc)


def plot_runner(runner, figure_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot(runner, axes)
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)


def run_experiment(name, model, train_set, valid_set, test_set, optimizer, scheduler=None, num_epochs=3, batch_size=64, log_iters=100, eval_interval=100):
    save_dir = RESULTS_DIR / name
    save_dir.mkdir(parents=True, exist_ok=True)

    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=int(np.max(train_set[1])) + 1)
    runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, batch_size=batch_size, scheduler=scheduler)
    runner.train(train_set, valid_set, num_epochs=num_epochs, log_iters=log_iters, eval_interval=eval_interval, save_dir=save_dir)

    best_model_path = save_dir / 'best_model.pickle'
    valid_loss, valid_acc = evaluate_model(model, valid_set[0], valid_set[1])
    test_loss, test_acc = evaluate_model(model, test_set[0], test_set[1])

    plot_runner(runner, save_dir / 'learning_curve.png')

    summary = {
        'name': name,
        'best_dev_score': float(getattr(runner, 'best_score', 0.0)),
        'final_valid_loss': valid_loss,
        'final_valid_acc': valid_acc,
        'final_test_loss': test_loss,
        'final_test_acc': test_acc,
        'best_model_path': str(best_model_path),
    }

    with open(save_dir / 'summary.pkl', 'wb') as f:
        pickle.dump(summary, f)

    print(f"[Summary] {name}: dev={valid_acc:.4f}, test={test_acc:.4f}, best={summary['best_dev_score']:.4f}")
    return summary


def run_mlp_baseline(data):
    model = nn.models.Model_MLP([data['train_flat'].shape[-1], 256, 10], 'ReLU', [1e-4, 1e-4])
    optimizer = nn.optimizer.SGD(init_lr=0.05, model=model)
    scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[300, 600, 900], gamma=0.5)
    return run_experiment(
        'mlp_baseline',
        model,
        [data['train_flat'], data['train_labels']],
        [data['valid_flat'], data['valid_labels']],
        [data['test_flat'], data['test_labels']],
        optimizer,
        scheduler=scheduler,
        num_epochs=1,
        batch_size=128,
        log_iters=5,
        eval_interval=10**9,
    )


def run_cnn_baseline(data):
    model = nn.models.Model_CNN(lambda_list=[1e-4, 1e-4, 1e-4, 1e-4])
    optimizer = nn.optimizer.SGD(init_lr=0.02, model=model)
    scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[300, 600, 900], gamma=0.5)
    return run_experiment(
        'cnn_baseline',
        model,
        [data['train_cnn'], data['train_labels']],
        [data['valid_cnn'], data['valid_labels']],
        [data['test_cnn'], data['test_labels']],
        optimizer,
        scheduler=scheduler,
        num_epochs=1,
        batch_size=128,
        log_iters=5,
        eval_interval=10**9,
    )


def run_optimization_experiment(data):
    model = nn.models.Model_CNN(lambda_list=[1e-4, 1e-4, 1e-4, 1e-4])
    optimizer = nn.optimizer.MomentGD(init_lr=0.02, model=model, mu=0.9)
    scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[300, 600, 900], gamma=0.5)
    return run_experiment(
        'cnn_momentum',
        model,
        [data['train_cnn'], data['train_labels']],
        [data['valid_cnn'], data['valid_labels']],
        [data['test_cnn'], data['test_labels']],
        optimizer,
        scheduler=scheduler,
        num_epochs=1,
        batch_size=128,
        log_iters=5,
        eval_interval=10**9,
    )


def run_regularization_experiment(data):
    results = []
    for name, lambdas in [
        ('cnn_reg_none', [None, None, None, None]),
        ('cnn_reg_mild', [1e-4, 1e-4, 1e-4, 1e-4]),
        ('cnn_reg_strong', [5e-4, 5e-4, 5e-4, 5e-4]),
    ]:
        model = nn.models.Model_CNN(lambda_list=lambdas)
        optimizer = nn.optimizer.SGD(init_lr=0.02, model=model)
        scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[300, 600, 900], gamma=0.5)
        results.append(run_experiment(
            name,
            model,
            [data['train_cnn'], data['train_labels']],
            [data['valid_cnn'], data['valid_labels']],
            [data['test_cnn'], data['test_labels']],
            optimizer,
            scheduler=scheduler,
            num_epochs=3,
            batch_size=64,
            log_iters=50,
            eval_interval=50,
        ))
    return results


def save_all_results(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / 'experiment_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    summary_path = RESULTS_DIR / 'experiment_results.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        for item in results:
            if isinstance(item, list):
                for sub_item in item:
                    f.write(f"{sub_item['name']}: dev={sub_item['final_valid_acc']:.4f}, test={sub_item['final_test_acc']:.4f}, best={sub_item['best_dev_score']:.4f}\n")
            else:
                f.write(f"{item['name']}: dev={item['final_valid_acc']:.4f}, test={item['final_test_acc']:.4f}, best={item['best_dev_score']:.4f}\n")


def main():
    data = load_mnist()
    results = []
    results.append(run_mlp_baseline(data))
    results.append(run_cnn_baseline(data))
    results.append(run_optimization_experiment(data))
    results.append(run_regularization_experiment(data))
    save_all_results(results)


if __name__ == '__main__':
    main()
