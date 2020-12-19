import gzip
from pathlib import Path
import struct

import numpy as np

current_dir = Path(__file__).parent


def load_dataset(path, is_label=False):
    with gzip.open(path) as f:
        magic_number, size = struct.unpack('>II', f.read(8))
        if is_label:
            return np.frombuffer(f.read(), dtype=np.dtype(np.uint8))
        else:
            rows, cols = struct.unpack('>II', f.read(8))
            data = np.frombuffer(f.read(), dtype=np.dtype(np.uint8))
            data = data.reshape((size, rows, cols))
            print(size, rows, cols)
            return data


def load_mnist():
    """Return shape: 10000 * 28 * 28
    """
    data_path = Path(current_dir, 'mnist')
    train_data = load_dataset(Path(data_path, 'train-images-idx3-ubyte.gz'))
    train_labels = load_dataset(Path(data_path, 'train-labels-idx1-ubyte.gz'),
                                is_label=True)
    test_data = load_dataset(Path(data_path, 't10k-images-idx3-ubyte.gz'))
    test_labels = load_dataset(Path(data_path, 't10k-labels-idx1-ubyte.gz'),
                               is_label=True)
    return train_data, train_labels, test_data, test_labels


def show(data):
    import matplotlib.pyplot as plt
    plt.imshow(data[0, :, :], cmap='gray')
    plt.show()


if __name__ == '__main__':
    show(load_mnist()[0])
