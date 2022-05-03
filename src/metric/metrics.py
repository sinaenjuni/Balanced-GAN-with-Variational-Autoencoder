import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from models.gan.embedded_generator import Generator
from datasets.imbalance_cifar import Imbalanced_CIFAR10 as Dataset
from torchvision.transforms import *
from torch.utils.data import DataLoader

_ = torch.manual_seed(123)

device = torch.device('cuda' if not torch.cuda.is_available() else 'cpu')
print(device)

fid = FrechetInceptionDistance(feature=64).to(device)

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

transforms = Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(15),
    Resize(64),
    ToTensor(),
    # Normalize(mean=[0.5], std=[0.5])
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
train_dataset = Dataset(root='~/data/',
                       train=True,
                       imb_factor=imb_factor,
                       download=True,
                       transform=transforms,
                        imb_type='ebgan')

from torch.utils.data import random_split
train_dataset, test_dataset = random_split(train_dataset, [6650, 2850], generator=torch.Generator().manual_seed(42))


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

img_ori = []
label_ori = []

for img, label in train_loader:
    img_ori.append(img)
    label_ori.append(label)
    # print(img.size())
    # print(label)
img_ori = ((torch.cat(img_ori)* 0.5 + 0.5) * 255.0).to(torch.uint8)
label_ori = torch.cat(label_ori)


# counts = {i : 0 for i in range(10)}
# for img, label in train_loader:
#     for l in label:
#         counts[l.item()] += 1

PATH = "/home/sin/git/ae/src/weights/evae/cifar10/g_0.pth"

g = Generator(image_size=image_size,
              image_channel=image_channel,
              std_channel=std_channel,
              latent_dim=latent_dim,
              num_class=num_class,
              norm='bn').to(device)

g.load_state_dict(torch.load(PATH))
noise = torch.randn((100, 128)).to(device)
label = (torch.ones(100) * 0).long().to(device)
img_gen = ((g(noise, label) * 0.5 + 0.5) * 255.0).to(torch.uint8)

fid.update(img_ori[label_ori==0], real=True)
fid.update(img_gen, real=False)
score = fid.compute()
print('>>FID: %.3f' % score)
print('-' * 50)
