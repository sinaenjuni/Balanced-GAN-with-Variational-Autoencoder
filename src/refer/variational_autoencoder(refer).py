# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from datasets.dataset import mnist


bs = 64
# MNIST Dataset
train_dataset = datasets.MNIST(root='../datasets/mnist/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='../datasets/mnist/', train=False, transform=transforms.ToTensor(), download=False)




# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

train_loader = mnist(image_size=28, train=True, batch_size=bs)

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
        return torch.tanh_(self.fc6(h))

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(Encoder, self).__init__()
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def forward(self, x):
        h = F.relu(self.fc1(x.view(-1, 784)))
        h = F.relu(self.fc2(h))
        mu = self.fc31(h)
        log_var = self.fc32(h)
        return self.sampling(mu, log_var), mu, log_var # mu, log_var

class Decoder(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(Decoder, self).__init__()
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

    def forward(self, x):
        h = F.relu(self.fc4(x))
        h = F.relu(self.fc5(h))
        h = self.fc6(h)
        return torch.tanh_(h)

class VAEE(nn.Module):
    def __init__(self):
        super(VAEE, self).__init__()
        self.encoder = Encoder(x_dim=784, h_dim1=512, h_dim2=256, z_dim=128)
        self.decoder = Decoder(x_dim=784, h_dim1=512, h_dim2=256, z_dim=128)

    def forward(self, x):
        encode, mu, log_var = self.encoder(x)
        decode = self.decoder(encode)
        return decode, mu, log_var

# build model
vae = VAE(x_dim=784, h_dim1=512, h_dim2=256, z_dim=128)
vaee = VAEE()

if torch.cuda.is_available():
    vae.cuda()
    vaee.cuda()

# optimizer = optim.Adam(vae.parameters())

optimizer = optim.Adam(vaee.parameters())

# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    BCE = F.mse_loss(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()

        # recon_batch, mu, log_var = vae(datasets)
        recon_batch, mu, log_var = vaee(data)


        loss = loss_function(recon_batch, data, mu, log_var)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def test():
    vae.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.cuda()
            # recon, mu, log_var = vae(datasets)
            recon, mu, log_var = vaee(data)

            # sum up batch loss
            test_loss += loss_function(recon, data, mu, log_var).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, 51):
    train(epoch)
    test()

    with torch.no_grad():
        z = torch.randn(64, 128).cuda()
        # sample = vae.decoder(z).cuda()
        sample = vaee.decoder(z)
        # save_image(sample.view(64, 1, 28, 28), './samples/sample_' + '.png')


        grid = make_grid(sample.view(64, 1, 28, 28).detach().cpu(), nrow=10, normalize=True)
        plt.imshow(grid.permute(1, 2, 0))
        plt.show()