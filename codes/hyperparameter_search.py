from pathlib import Path
import pickle

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / 'saved_models'


def load_results():
    results_path = RESULTS_DIR / 'experiment_results.pkl'
    if not results_path.exists():
        raise FileNotFoundError(f'{results_path} does not exist. Run test_train.py first.')
    with open(results_path, 'rb') as f:
        return pickle.load(f)


def flatten_results(results):
    flattened = []
    for item in results:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened


def print_summary(results):
    print('name\tdev_acc\ttest_acc\tbest_dev')
    for item in flatten_results(results):
        print(f"{item['name']}\t{item['final_valid_acc']:.4f}\t{item['final_test_acc']:.4f}\t{item['best_dev_score']:.4f}")


if __name__ == '__main__':
    print_summary(load_results())
