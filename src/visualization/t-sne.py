import torch
import numpy as np
from datasets.dataset import cifar
from models.modules.encoder_module import Encoder_module
from models.modules.decoder_module import Decoder_module
from sklearn.manifold import TSNE
import matplotlib.pylab as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

SAVA_PATH = f"../../weights/ae/cifar10/"


target_epoch = 99

batch_size = 128
image_size = 32
image_channel = 3
std_channel = 64
latent_dim = 128

encoder = Encoder_module(image_size=image_size,
                         image_channel=image_channel,
                         std_channel=std_channel,
                         latent_dim=latent_dim).to(device)
encoder.load_state_dict(torch.load(SAVA_PATH+f"encoder_{target_epoch}.pth"))

decoder = Decoder_module(image_size=image_size,
                         image_channel=image_channel,
                         std_channel=std_channel,
                         latent_dim=latent_dim).to(device)
decoder.load_state_dict(torch.load(SAVA_PATH+f"decoder_{target_epoch}.pth"))



train_loader = cifar(image_size=32, train=True, batch_size=128)
test_loader  = cifar(image_size=32, train=False, batch_size=128)


latents = np.empty((0, 128))
labels = np.empty((0,))
for idx, (image, label) in enumerate(train_loader):
    image = image.to(device)
    output = encoder(image).detach().cpu().numpy()

    latents = np.append(latents, output, axis=0)
    labels = np.append(labels, label)


tsne = TSNE(n_components=2, random_state=0, n_iter=2000, verbose=1)
tsne_results = tsne.fit_transform(latents)
print(tsne_results)


target_ids = [0,1,2,3,4,5,6,7,8,9]
plt.figure(figsize=(6, 5))
colors = {0:'r', 1:'g', 2:'b', 3:'c', 4:'m', 5:'y', 6:'k', 7:'lime', 8:'orange', 9:'purple'}
for i in target_ids:
    plt.scatter(tsne_results[labels == i, 0], tsne_results[labels== i, 1], c=colors[i], label=i, alpha=.3)
plt.legend()
plt.show()