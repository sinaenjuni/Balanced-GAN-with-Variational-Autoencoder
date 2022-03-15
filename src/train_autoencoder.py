import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.utils import make_grid
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torchvision.transforms import Compose, Resize, Normalize, ToTensor

from models.encoder import Encoder
from models.decoder import Decoder

# import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

SAVE_PATH = "../weights/ae/"

# Hyper-parameters
batch_size = 128
num_epoch = 100
image_size = 32
image_channel = 3
std_channel = 64
latent_dim = 128
learning_rate = 0.0002
beta1 = 0.5
beta2 = 0.9

transform = Compose([Resize(image_size),
                     ToTensor(),
                     Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

dataset = CIFAR10

train_dataset = dataset(root='../data/', download=True, train=True, transform=transform)
test_dataset = dataset(root='../data/', download=True, train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


encoder = Encoder(image_size=image_size,
                  image_channel=image_channel,
                  std_channel=std_channel,
                  latent_dim=latent_dim).to(device)

decoder = Decoder(image_size=image_size,
                  image_channel=image_channel,
                  std_channel=std_channel,
                  latent_dim=latent_dim).to(device)

criterion = nn.MSELoss()
optimizer = Adam([{'params':encoder.parameters(),
                              'params': decoder.parameters()}],
                             lr=learning_rate,
                             weight_decay=1e-5,
                             betas=(beta1, beta2))

for epoch in range(num_epoch):
    train_loss = 0
    for i, (image, labels) in enumerate(train_loader):
        image = image.to(device)

        encode = encoder(image)
        decode = decoder(encode)

        loss = criterion(decode, image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch: {train_loss}")
    grid = make_grid(decode.detach().cpu(), nrow=10, normalize=True)
    plt.imshow(grid.permute(1,2,0))
    plt.show()

    # if not os.path.exists(SAVE_PATH):
    #     os.makedirs(SAVE_PATH)




