import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.utils import make_grid

from models.gan.discriminator_sam1 import Discriminator
from models.modules.decoder_module import Decoder_module
from datasets.dataset import mnist

import matplotlib.pyplot as plt



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
learning_rate = 2e-4
beta1 = 0.5
beta2 = 0.9
dataset = mnist
image_size = 64
image_channel = 1

SAVE_PATH = f"../weights/gan/{dataset.__name__}/"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

train_loader = dataset(image_size=image_size, train=True, batch_size=128)
test_loader  = dataset(image_size=image_size, train=False, batch_size=128)

D = Discriminator(image_size=image_size,
                  image_channel=image_channel,
                  std_channel=std_channel,
                  latent_dim=latent_dim).to(device)

G = Decoder_module(image_size=image_size,
                   image_channel=image_channel,
                   std_channel=std_channel,
                   latent_dim=latent_dim).to(device)


bce_loss = nn.BCELoss()
d_optimizer = Adam(D.parameters(),
                 lr=learning_rate,
                 weight_decay=1e-5,
                 betas=(beta1, beta2))
g_optimizer = Adam(G.parameters(),
                 lr=learning_rate,
                 weight_decay=1e-5,
                 betas=(beta1, beta2))


fixed_noise = torch.randn(100, latent_dim).to(device)

for epoch in range(num_epoch):
    train_loss = 0
    for i, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)

        _batch = label.size(0)
        real_label = torch.ones(_batch, 1).to(device)
        fake_label = torch.zeros(_batch, 1).to(device)
        noise = torch.randn(_batch, latent_dim).to(device)

        fake_image = G(noise)

        ################################################# Train Generator
        d_optimizer.zero_grad()
        d_output_real = D(image)
        d_loss_real = bce_loss(d_output_real, real_label)

        d_output_fake = D(fake_image.detach())
        d_loss_fake = bce_loss(d_output_fake, fake_label)

        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        d_loss.backward()
        d_optimizer.step()


        ################################################# Train Discriminator
        g_optimizer.zero_grad()
        g_output = D(fake_image)
        g_loss = bce_loss(g_output, real_label)

        g_loss.backward()
        g_optimizer.step()


    print(f"Epoch: {epoch+1}/{num_epoch}, "
          f"D Score: {d_loss.item()/len(train_loader):.4f}"
          f"G Score: {g_loss.item()/len(train_loader):.4f}")

    with torch.no_grad():
        G.eval()
        fixed_vector_output = G(fixed_noise)
        grid = make_grid(fixed_vector_output.detach().cpu(), nrow=10, normalize=True)
        plt.imshow(grid.permute(1,2,0))
        plt.show()
    # torch.save(encoder.state_dict(), SAVE_PATH + f"encoder_{epoch+1}.pth")
    # torch.save(decoder.state_dict(), SAVE_PATH + f"decoder_{epoch+1}.pth")


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



    

