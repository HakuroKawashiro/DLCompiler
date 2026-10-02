import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        self.transforms = transforms
        train_names = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        test_names = ['test_batch']
        names = train_names if train else test_names
        dicts = []
        for name in names:
            with open(os.path.join(base_folder, name), 'rb') as f:
                dicts.append(pickle.load(f, encoding='bytes'))
        self.X = np.concatenate([d[b'data'] for d in dicts], axis=0).reshape(-1, 3, 32, 32)
        self.X = self.X / 255.0
        self.y = np.concatenate([d[b'labels'] for d in dicts], axis=0)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        data = np.transpose(np.reshape(self.X[index], (-1, 32, 32)), (1, 2, 0))
        data = np.reshape(np.transpose(self.apply_transforms(data), (2, 0, 1)), (-1, 3, 32, 32))
        if data.shape[0] == 1:
            data = np.reshape(data, (3, 32, 32))
        label = self.y[index]
        return data, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION
