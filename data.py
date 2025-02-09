import numpy as np
import jax
import torchvision.datasets as datasets
import torch.utils.data.dataloader as DataLoader
from hparam import Hyperparameters 
hps = Hyperparameters()

# For transformation
def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    if img.max() > 1:
        img = img / 255. * 2. - 1.
    return img

# General numpy collate function for JAX
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

class Dataset:
    """
    Download the dataset
    Prepare dataset as an input of the DataLoader
    """
    def __init__(self, download=True):
        self.data_path = hps.data_path
        self.download = download
        self.dataset = None
        self.train_dataset = self.prepare_data(hps.data_name, train=True)
        self.test_dataset = self.prepare_data(hps.data_name, train=False)

    def train_loader(self):
        return DataLoader(self.train_dataset,
                          batch_size=hps.batch_size, 
                          shuffle=True, 
                          pin_memory=True,
                          drop_last=True,
                          collate_fn=numpy_collate)

    def test_loader(self):
        return DataLoader(self.test_dataset,
                          batch_size=hps.batch_size, 
                          shuffle=False, 
                          drop_last=False,
                          collate_fn=numpy_collate)

    def prepare_data(self, data_name, train=True):
        
        if data_name == 'MNIST':
            self.dataset = datasets.MNIST
        elif data_name == 'CIFAR10':
            self.dataset = datasets.CIFAR10
        else:
            raise ValueError('Dataset name is not valid')

        def dataset_(train):
            return self.dataset(root=self.data_path, 
                                train=train,
                                transform=image_to_numpy, 
                                download=self.download)

        return dataset_(train)

class Preprocessing():
    def __init__(self):
        pass
    """
    Preprocess data
    - resize(scaling)
    - normalize RGB
    - augmentation if needed
    """

    