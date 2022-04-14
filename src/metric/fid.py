import numpy as np
from scipy.linalg import sqrtm

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import *

from datasets.imbalance_fashion_mnist import Imbalanced_FashionMNIST
from datasets.imbalance_cifar import Imbalanced_CIFAR10

from datasets.sampler import BalancedSampler
from metric.inception_net import InceptionV3


batch_size = 128
imb_factor = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transforms = Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(15),
    Resize(299),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = Imbalanced_CIFAR10(root='~/data/',
                       train=True,
                       imb_factor=imb_factor,
                       download=True,
                       transform=transforms)

test_dataset = Imbalanced_CIFAR10(root='~/data/',
                       train=False,
                       imb_factor=imb_factor,
                       download=True,
                       transform=transforms)


# sampler = BalancedSampler(train_dataset, retain_epoch_size=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
fixed_image, fixed_label = iter(train_loader).__next__()

sample_transform = Compose([
    Resize(299),
])


inputs = torch.randn((200, 3, 299, 299)).to(device)
model = InceptionV3(resize_input=False, normalize_input=False).to(device)


real_image_features = []
gen_image_features = []

for idx, (image, label) in enumerate(train_loader):
    real_image = image.to(device)
    real_image_feature, real_image_logit = model(real_image)

    real_image_features.append(real_image_feature)


real_image_features = torch.cat(real_image_features, 0)

calculate_fid(real_image_features,
              real_image_features)

# %% --------------------------------------- Define FID ----------------------------------------------------------------
# Reference: https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
# calculate frechet inception distance
def calculate_fid(real_image_features, gen_image_features):
    # calculate activations
    act1 = real_image_features
    act2 = gen_image_features
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid