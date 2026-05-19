import gzip
from pathlib import Path
from struct import unpack

import mynn as nn
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'dataset' / 'MNIST'
MODEL_PATH = BASE_DIR / 'saved_models' / 'mlp_baseline' / 'best_model.pickle'
MODEL_TYPE = 'mlp'


def load_test_data(model_type):
    test_images_path = DATA_DIR / 't10k-images-idx3-ubyte.gz'
    test_labels_path = DATA_DIR / 't10k-labels-idx1-ubyte.gz'

    with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

    with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

    test_imgs = test_imgs.astype(np.float32) / 255.0
    if model_type == 'cnn':
        test_imgs = test_imgs.reshape(-1, 1, 28, 28)
    return test_imgs, test_labs


if MODEL_TYPE == 'cnn':
    model = nn.models.Model_CNN()
else:
    model = nn.models.Model_MLP()

model.load_model(MODEL_PATH)

test_imgs, test_labs = load_test_data(MODEL_TYPE)
logits = model(test_imgs)
print(nn.metric.accuracy(logits, test_labs))
