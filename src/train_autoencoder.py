import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from models.modules.encoder import Encoder
from models.modules.decoder import Decoder
from dataset import mnist

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
learning_rate = 0.0002
beta1 = 0.5
beta2 = 0.9
dataset = mnist
image_size = 64
image_channel = 1

SAVE_PATH = f"../weights/ae/{dataset.__name__}/"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

train_loader = dataset(image_size=image_size, train=True, batch_size=128)
test_loader  = dataset(image_size=image_size, train=False, batch_size=128)

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
    for i, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)

        encode = encoder(image)
        decode = decoder(encode)
        loss = criterion(decode, image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch: {epoch+1}/{num_epoch}, Loss: {train_loss/len(train_loader)}")
    torch.save(encoder.state_dict(), SAVE_PATH + f"encoder_{epoch+1}.pth")
    torch.save(decoder.state_dict(), SAVE_PATH + f"decoder_{epoch+1}.pth")


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



    

