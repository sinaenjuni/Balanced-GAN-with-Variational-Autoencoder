import torch
torch.manual_seed(123)
from models.gan.embedded_generator import Generator
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import *
from datasets.imbalance_cifar import Imbalanced_CIFAR10 as Dataset

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

li_test_dataset = list(test_dataset)

import numpy as np
li_test_dataset = np.array(li_test_dataset)

li_test_dataset[0,0].mean()
tensor(0.4260)
li_test_dataset[0,0].std()
tensor(0.1736)

g = Generator(image_size=image_size,
              image_channel=image_channel,
              std_channel=std_channel,
              latent_dim=latent_dim,
              num_class=num_class,
              norm='bn').to(device)


_batch = 1000

noise = torch.randn(_batch, latent_dim).to(device)



fid = FrechetInceptionDistance(feature=2048)

# generate two slightly overlapping image intensity distributions
imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
fid.update(imgs_dist1, real=True)
fid.update(imgs_dist2, real=False)
score = fid.compute()
print(score)

