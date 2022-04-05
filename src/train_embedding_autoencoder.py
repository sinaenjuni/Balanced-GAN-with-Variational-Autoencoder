import sys
sys.path.append(r'/home/sin/git/ae/src/')

import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from models.modules.encoder import Encoder
from models.modules.decoder import Decoder
from models.modules.embedding import Label_embedding_model
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
num_classes = 10

dataset = mnist
image_size = 64
image_channel = 1

SAVE_PATH = f"../weights/embedding_ae/{dataset.__name__}/"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

train_loader = dataset(image_size=image_size, train=True, batch_size=batch_size)
test_loader  = dataset(image_size=image_size, train=False, batch_size=batch_size)

fixed_image = next(iter(test_loader))[0].to(device)
fixed_label = next(iter(test_loader))[1].to(device)



def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, std=0.02)
        # nn.init.normal_(m.bias.data, std=0.02)
    if isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data, std=0.02)
        # nn.init.normal_(m.bias.data, std=0.02)

encoder = Encoder(image_size=image_size,
                  image_channel=image_channel,
                  std_channel=std_channel,
                  latent_dim=latent_dim).to(device).apply(initialize_weights)

decoder = Decoder(image_size=image_size,
                  image_channel=image_channel,
                  std_channel=std_channel,
                  latent_dim=latent_dim).to(device).apply(initialize_weights)

embedding = Label_embedding_model(num_classes=num_classes,
                                  latent_dim=latent_dim).to(device)

criterion = nn.MSELoss()
optimizer = Adam([{'params': encoder.parameters(),
                   'params': embedding.parameters(),
                   'params': decoder.parameters()
                   }],
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

    print(f"Epoch: {epoch+1}/{num_epoch}, Loss: {train_loss/len(train_loader)}")
    torch.save(encoder.state_dict(),         SAVE_PATH + f"encoder_{epoch+1}.pth")
    torch.save(embedding.state_dict(),       SAVE_PATH + f"embedding_{epoch+1}.pth")
    torch.save(decoder.state_dict(),         SAVE_PATH + f"decoder_{epoch+1}.pth")

# #################### Save numpy ###################
# test_encode_results = np.empty((0,128))
# test_embedding_encode_results = np.empty((0, 128))
# test_decode_results = np.empty((0, image_channel, image_size, image_size))
# for idx, (image, labels) in enumerate(test_loader):
#     image = image.to(device)
#     labels = labels.to(device)
#     with torch.no_grad():
#         encoder.eval()
#         decoder.eval()
#
#         encode = encoder(image)
#         embedded_encode = embedding(encode, labels)
#         decode = decoder(embedded_encode)
#
#     test_encode_results = np.append(test_encode_results, encode.detach().cpu().numpy(), axis=0)
#     test_embedding_results = np.append(test_embedding_encode_results, embedded_encode.detach().cpu().numpy(), axis=0)
#     test_decode_results = np.append(test_decode_results, decode.detach().cpu().numpy(), axis=0)
#
#     print(f"Epoch: {idx+1}/{len(test_loader)}")
#
# np.save(SAVE_PATH + "test_encode_results.npy", test_encode_results)
# np.save(SAVE_PATH + "test_embedding_encode_results.npy", test_embedding_encode_results)
# np.save(SAVE_PATH + "test_decode_results.npy", test_decode_results)

# grid = make_grid(decoder(torch.tensor(test_encode_results).float().to(device))[:100], nrow=10, normalize=True)
# plt.imshow(grid.permute(1, 2, 0))
# plt.show()
#
# decoder(torch.tensor(test_encode_results).float().to(device))
# grid = make_grid(torch.tensor(test_decode_results[:100]), nrow=10, normalize=True)
# plt.imshow(grid.permute(1, 2, 0))
# plt.show()
#
#
#
# test_labels = np.array(test_loader.dataset.test_labels)
#
# tsne_encode = TSNE(n_components=2, random_state=0, n_iter=2000, verbose=1)
# tsne_encode_results = tsne_encode.fit_transform(test_encode_results)
# print(tsne_encode_results)
#
# tsne_embedding_encode = TSNE(n_components=2, random_state=0, n_iter=2000, verbose=1)
# tsne_embedding_results = tsne_embedding_encode.fit_transform(test_embedding_results)
# print(tsne_embedding_results)
#
#
# target_ids = [0,1,2,3,4,5,6,7,8,9]
# plt.figure(figsize=(6, 5))
# colors = {0:'r', 1:'g', 2:'b', 3:'c', 4:'m', 5:'y', 6:'k', 7:'lime', 8:'orange', 9:'purple'}
# for i in target_ids:
#     plt.scatter(tsne_encode_results[test_labels==i, 0], tsne_encode_results[test_labels==i, 1], c=colors[i], label=i, alpha=.3)
# plt.legend()
# plt.show()
#
# plt.figure(figsize=(6, 5))
# colors = {0:'r', 1:'g', 2:'b', 3:'c', 4:'m', 5:'y', 6:'k', 7:'lime', 8:'orange', 9:'purple'}
# for i in target_ids:
#     plt.scatter(tsne_embedding_results[test_labels==i, 0], tsne_embedding_results[test_labels==i, 1], c=colors[i], label=i, alpha=.3)
# plt.legend()
# plt.show()




    

