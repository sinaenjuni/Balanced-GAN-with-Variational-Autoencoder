import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from PIL import Image

from torchvision.utils import make_grid
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, ToPILImage

# import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# Hyper-parameters
batch_size = 128


transform = Compose([ToTensor(),
                     Normalize([0.5], [0.5])])

dataset = MNIST

train_dataset = dataset(root='../data/', download=True, train=True, transform=transform)
test_dataset = dataset(root='../data/', download=True, train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)



# print(all_train_data.size())
# print(transform(all_train_data).size())

# print(transform(train_dataset.train_data[0]))
# Image.fromarray(train_dataset.train_data[0].numpy(), mode='L')

# for i in train_dataset.train_data[:1]:
#     print(i)
#     print(t(i))

# all_train_dataset = [transform(i) for i in train_dataset.train_data]
# all_test_dataset = (test_dataset.test_data)
#
# print((all_train_dataset))
# print(all_test_dataset)


