import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.utils import make_grid

from models.encoder import Encoder
from models.decoder import Decoder
from dataset import mnist

# import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

SAVE_PATH = "../weights/ae/"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Hyper-parameters
batch_size = 128
num_epoch = 100
image_size = 32
image_channel = 1
std_channel = 64
latent_dim = 128
learning_rate = 0.0002
beta1 = 0.5
beta2 = 0.9


train_loader = mnist(image_size=32, train=True, batch_size=128)
test_loader  = mnist(image_size=32, train=False, batch_size=128)

fixed_latent_vector = torch.randn((100, 128)).to(device)


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

    decoder.eval()
    fixed_vector_output = decoder(fixed_latent_vector)
    grid = make_grid(fixed_vector_output.detach().cpu(), nrow=10, normalize=True)
    plt.imshow(grid.permute(1,2,0))
    plt.show()


    

