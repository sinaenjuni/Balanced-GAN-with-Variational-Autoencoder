import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.utils import make_grid

from models.encoder import Encoder
from models.decoder import Decoder
from models.embedding import Label_embedding_model
from dataset import mnist, cifar

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


SAVE_PATH = "../weights/embedding_ae/cifar10/"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

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
num_classes = 10

train_loader = cifar(image_size=32, train=True, batch_size=128)
test_loader  = cifar(image_size=32, train=False, batch_size=128)



fixed_test_dataset = next(iter(test_loader))[0].to(device)


encoder = Encoder(image_size=image_size,
                  image_channel=image_channel,
                  std_channel=std_channel,
                  latent_dim=latent_dim).to(device)

decoder = Decoder(image_size=image_size,
                  image_channel=image_channel,
                  std_channel=std_channel,
                  latent_dim=latent_dim).to(device)

embedding = Label_embedding_model(num_classes=num_classes,
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
        labels = labels.to(device)

        encode = encoder(image)
        label_embedding = embedding(encode, labels)
        decode = decoder(label_embedding)
        loss = criterion(decode, image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch: {train_loss/len(train_loader)}")

    with torch.no_grad():
        decoder.eval()
        fixed_vector_output = decoder(encoder(fixed_test_dataset))
        grid = make_grid(fixed_vector_output.detach().cpu(), nrow=10, normalize=True)
        plt.imshow(grid.permute(1,2,0))
        plt.show()

    torch.save(encoder.state_dict(), SAVE_PATH + f"encoder_{epoch}.pth")
    torch.save(decoder.state_dict(), SAVE_PATH + f"decoder_{epoch}.pth")


    

