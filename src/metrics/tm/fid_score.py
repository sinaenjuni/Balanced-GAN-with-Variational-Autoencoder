import torch
torch.manual_seed(123)
from models.gan.embedded_generator import Generator
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import *
from datasets.imbalance_cifar import Imbalanced_CIFAR10 as Dataset
import numpy as np

from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# Hyper-parameters
batch_size = 128
eae_num_epoch = 30
gan_num_epoch = 100
std_channel = 64
latent_dim = 128
num_class = 10
imb_factor = 0.01
learning_rate = 2e-4
train_ratio = 5
gp_weight = 10
beta1 = 0.5
beta2 = 0.9
dataset = 'ebgan_cifar10'
image_size = 64
image_channel = 3
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

resize299 = Resize(299)

def getDataOri():
    transforms = Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        Resize(64),
        ToTensor(),
        # Normalize(mean=[0.5], std=[0.5])
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_dataset = Dataset(root='~/data/',
                           train=False,
                           imb_factor=imb_factor,
                           download=True,
                           transform=transforms,
                            imb_type='ebgan')
    images = [tensor[0] for tensor in test_dataset]
    labels = [tensor[1] for tensor in test_dataset]

    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels

def getTargetDataOri(images, labels, idx):
    idx_ = torch.where(labels == idx)
    return images[idx_].to(device)

def showImages(images):
    grid_image_ori = make_grid(images, normalize=True, nrow=30).permute(1, 2, 0).cpu().numpy()
    plt.imshow(grid_image_ori)
    plt.show()

def getGenerator():
    return Generator(image_size=image_size,
                  image_channel=image_channel,
                  std_channel=std_channel,
                  latent_dim=latent_dim,
                  num_class=num_class,
                  norm='bn').to(device)



def setGenerator(g, epoch=99):
    g.load_state_dict(torch.load(f"/home/dblab/git/VAE-GAN/src/save_files/evae/ebgan_cifar10/weights/g_{epoch}.pth"))


def getNoiseAndLabels(batch, label):
    noise = torch.randn(batch, latent_dim)
    noise_labels = torch.ones(batch).to(torch.long) * label
    return noise.to(device), noise_labels.to(device)

def denorm(images):
    return ((images * .5 + .5) * 255).to(torch.uint8)
    # images_denorm = (((images * .5) + .5) * 255)
    # return torch.clamp(images_denorm, min=0, max=1).to(torch.uint8)

G = getGenerator()
setGenerator(G, 99)

images_ori, labels_ori = getDataOri()

target = 0
images_ori = getTargetDataOri(images_ori, labels_ori, target)

noise, noise_labels = getNoiseAndLabels(1000, target)
images_gen = G(noise, noise_labels)

images_ori = denorm(images_ori)
images_gen = denorm(images_gen)


showImages(make_grid(images_ori))
showImages(make_grid(images_gen))






fid = FrechetInceptionDistance(feature=2048).to(device)
fid.update(images_ori, real=True)
fid.update(images_gen, real=False)
score = fid.compute()
print(score)


# generate two slightly overlapping image intensity distributions
imgs_dist1 = torch.randint(0, 200, (100, 3, 64, 64), dtype=torch.uint8)
imgs_dist2 = torch.randint(100, 255, (100, 3, 64, 64), dtype=torch.uint8)
fid.update(imgs_dist1, real=True)
fid.update(imgs_dist2, real=False)
score = fid.compute()
print(score)


imgs_dist1 = resize299(imgs_dist1)
imgs_dist2 = resize299(imgs_dist2)
fid.update(imgs_dist1, real=True)
fid.update(imgs_dist2, real=False)
score = fid.compute()
print(score)