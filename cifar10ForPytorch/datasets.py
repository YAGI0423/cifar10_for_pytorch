import numpy as np
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset

from torch import Tensor


class Cifar10Dataset(Dataset):
    '''
    classes = {
        0: 'plane',
        1: 'car',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck',
    }
    '''
    def __init__(self, is_train: bool=True,
                 flatten: bool=False, normalize: bool=False, root: str='./cifar10ForPytorch/cifar10') -> None:
        
        self.x, self.y = self.__getCifar10(root=root, is_train=is_train)
        
        if flatten:
            self.x = self.__flatten(self.x)
        if normalize:
            self.x = self.__minMax_normalize(self.x)

    def __len__(self) -> int:
        return self.x.__len__()

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        return self.x[index], self.y[index]
    
    def __getCifar10(self, root: str, is_train: bool) -> tuple[Tensor, Tensor]:
        cifar = CIFAR10(root, train=is_train, download=True)
        return Tensor(cifar.data.astype(np.float32)), Tensor(cifar.targets).long()
    
    def __flatten(self, x) -> Tensor:
        batch, height, weight, channel = x.shape
        return x.reshape(batch, height * weight * channel)

    def __minMax_normalize(self, x) -> Tensor:
        return x / x.max()