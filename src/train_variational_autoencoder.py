import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.utils import make_grid

from models.ae.autoencoder import VAE
from datasets.dataset import cifar

# import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# %% --------------------------------------- Fix Seeds -----------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.benchmark = True



# Hyper-parameters
batch_size = 128
num_epoch = 30
std_channel = 64
latent_dim = 128
learning_rate = 1e-3
beta1 = 0.5
beta2 = 0.9
dataset = cifar
image_size = 64
image_channel = 3

SAVE_PATH = f"../weights/var_ae/{dataset.__name__}/"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

train_loader = dataset(image_size=image_size, train=True, batch_size=batch_size)
test_loader  = dataset(image_size=image_size, train=False, batch_size=10000)

test_images = test_loader.__iter__().next()[0]
test_labels = test_loader.__iter__().next()[1]
target_idx = np.array([[np.where(test_labels==idx)[0][iter]
                        for idx in np.unique(test_labels)]
                       for iter in range(10)]).ravel()
test_images = test_loader.__iter__().next()[0][target_idx].to(device)
test_labels = test_loader.__iter__().next()[1][target_idx].to(device)


vae = VAE(image_size=image_size,
                  image_channel=image_channel,
                  std_channel=std_channel,
                  latent_dim=latent_dim).to(device)


# BCELoss = nn.MSELoss(reduction='sum')
# BCELoss = nn.MSELoss(reduction='sum')
# KLDLoss = lambda mu, log_var: -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
# optimizer = Adam([{'params':encoder.parameters(),
#                    'params': decoder.parameters()}],
#                              lr=learning_rate,
#                              weight_decay=1e-5,
#                              betas=(beta1, beta2))
optimizer = Adam(vae.parameters(), lr=learning_rate,
                             weight_decay=1e-3,
                             betas=(beta1, beta2))
# optimizer = Adam(vae.parameters())

def loss_function(recon_x, x, mu, log_var):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    # KLD = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

    return BCE + KLD

for epoch in range(num_epoch):
    train_loss = 0
    for i, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)

        decode, mu, log_var = vae(image)

        # bceloss = BCELoss(decode, image)
        # kldloss = KLDLoss(mu, log_var)
        # loss = bceloss + kldloss

        loss = loss_function(decode, image, mu, log_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch: {epoch+1}/{num_epoch}, Loss: {train_loss/len(train_loader)}")
    # torch.save(encoder.state_dict(), SAVE_PATH + f"encoder_{epoch+1}.pth")
    # torch.save(decoder.state_dict(), SAVE_PATH + f"decoder_{epoch+1}.pth")
    with torch.no_grad():
        vae.eval()
        z = torch.randn(64, 128).cuda()
        decode = vae.decoder(z)
        # encode, mu, log_var = encoder(test_images)
        # decode = decoder(encode)
        grid = make_grid(decode.detach().cpu(), nrow=10, normalize=True)
        plt.imshow(grid.permute(1,2,0))
        plt.show()



# #################### Save numpy ###################
# test_encode_results = np.empty((0,128))
# test_decode_results = np.empty((0, image_channel, image_size, image_size))
# for idx, (image, labels) in enumerate(test_loader):
#     image = image.to(device)
#     labels = labels.to(device)
#     with torch.no_grad():
#         encoder.eval()
#         decoder.eval()
#
#         encode = encoder(image)
#         decode = decoder(encode)
#
#     test_encode_results = np.append(test_encode_results, encode.detach().cpu().numpy(), axis=0)
#     test_decode_results = np.append(test_decode_results, decode.detach().cpu().numpy(), axis=0)
#
#     print(f"Epoch: {idx+1}/{len(test_loader)}")


# with torch.no_grad():
#     decoder.eval()
#     fixed_vector_output = decoder(encoder(fixed_test_dataset))
#     grid = make_grid(fixed_vector_output.detach().cpu(), nrow=10, normalize=True)
#     plt.imshow(grid.permute(1,2,0))
#     plt.show()
#

