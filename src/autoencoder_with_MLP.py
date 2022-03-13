import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.utils import make_grid
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import Compose, Resize, Normalize, ToTensor

# import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



transform = Compose([ToTensor(),
                      Normalize([0.5], [0.5]),
                     ])

train_cifar10 = CIFAR10(root='../data/', download=False, train=True, transform=transform)

print(train_cifar10[0])
# print("etset")