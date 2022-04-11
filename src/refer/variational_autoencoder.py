import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torchvision.utils import make_grid

from datasets.dataset import mnist

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


train_loader = mnist(image_size=28, train=True, batch_size=batch_size)
test_loader  = mnist(image_size=28, train=False, batch_size=batch_size)

fixed_latent_vector = torch.randn((100, 128)).to(device)


class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()

        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var


# build model
vae = VAE(x_dim=784, h_dim1=512, h_dim2=256, z_dim=128).to(device)
criterion = nn.MSELoss()

optimizer = Adam(vae.parameters())

def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

for epoch in range(num_epoch):
    train_loss = 0
    for i, (image, labels) in enumerate(train_loader):
        image = image.to(device)


        recon_batch, mu, log_var = vae(image)
        loss = loss_function(recon_batch, image, mu, log_var)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch: {train_loss/ len(train_loader)}")
    with torch.no_grad():
        z = torch.randn(64, 128).cuda()
        sample = vae.decoder(z).cuda()

        grid = make_grid(sample.view(64, 1, 28, 28).detach().cpu(), nrow=10, normalize=True)
        plt.imshow(grid.permute(1, 2, 0))
        plt.show()

        # decoder.eval()
        # fixed_vector_output = decoder(fixed_latent_vector)
        # grid = make_grid(fixed_vector_output.detach().cpu(), nrow=10, normalize=True)
        # plt.imshow(grid.permute(1,2,0))
        # plt.show()


    

