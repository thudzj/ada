import gzip
import os
from urllib.parse import urljoin

import numpy as np

from .dataset import make_dataset
from . import util


class USPS():
    """USPS handwritten digits.
    Homepage: http://statweb.stanford.edu/~hastie/ElemStatLearn/data.html
    Images are 16x16 grayscale images in the range [0, 1].
    """

    base_url = 'http://statweb.stanford.edu/~hastie/ElemStatLearn/datasets/'

    data_files = {
        'train': 'zip.train.gz',
        'test': 'zip.test.gz'
        }

    num_classes = 10

    def __init__(self, path=None, download=True):
        self.image_shape = (16, 16, 1)
        self.label_shape = ()
        self.path = path
        self.download()
        self._load_datasets()
    def get_path(self, *args):
        return os.path.join(self.path, 'usps', *args)

    def download(self):
        data_dir = self.get_path()
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        for filename in self.data_files.values():
            path = self.get_path(filename)
            if not os.path.exists(path):
                url = urljoin(self.base_url, filename)
                util.maybe_download(url, path)

    def _load_datasets(self):
        abspaths = {name: self.get_path(path)
                    for name, path in self.data_files.items()}
        train_images, train_labels = self._read_datafile(abspaths['train'])
        test_images, test_labels = self._read_datafile(abspaths['test'])
        self.train = make_dataset(train_images, train_labels)
        self.test = make_dataset(test_images, test_labels)

    def _read_datafile(self, path):
        """Read the proprietary USPS digits data file."""
        labels, images = [], []
        with gzip.GzipFile(path) as f:
            for line in f:
                vals = line.strip().split()
                labels.append(float(vals[0]))
                images.append([float(val) for val in vals[1:]])
        labels = np.array(labels, dtype=np.int32)
        labels[labels == 10] = 0  # fix weird 0 labels
        images = np.array(images, dtype=np.float32).reshape(-1, 16, 16, 1)
        #images = (images + 1) / 2
        return images, labels
