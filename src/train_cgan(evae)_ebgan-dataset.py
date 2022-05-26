import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import make_grid
from torchvision.transforms import *


from models.gan.discriminator import Discriminator
from models.gan.embedded_generator import Generator
from models.ae.embedding_variational_autoencoder import EVAE
from models.modules.weights_initialize_method import initialize_weights

# from datasets.imbalance_fashion_mnist import Imbalanced_FashionMNIST
from datasets.imbalance_cifar import Imbalanced_CIFAR10 as Dataset
# from torchvision.datasets import FashionMNIST
from datasets.sampler import BalancedSampler

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

BASE_PATH = f'save_files/evae/{dataset}/'
IMG_PATH = f'{dataset}/images/'
WEIGHT_PATH = f'{dataset}/weights/'
if not os.path.exists(IMG_PATH):
    os.makedirs(IMG_PATH)
if not os.path.exists(WEIGHT_PATH):
    os.makedirs(WEIGHT_PATH)


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


# test_dataset = Dataset(root='~/data/',
#                        train=False,
#                        imb_factor=imb_factor,
#                        download=True,
#                        transform=transforms)


# sampler = BalancedSampler(train_dataset, retain_epoch_size=False)
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=10000, shuffle=False)

fixed_image, fixed_label = iter(test_loader).__next__()
target_idx = np.array([np.where(fixed_label == i)[0][:10] for i in fixed_label.unique()]).ravel()

fixed_noise = torch.randn(10, latent_dim).to(device).repeat(10, 1)
fixed_noise_label = torch.tensor([[i//10] for i in range(100)]).to(device)
# fixed_noise_label = torch.zeros(100, 10).scatter_(1, index, 1).to(device)

# grid = make_grid(fixed_image[target_idx], normalize=True, nrow=10)
# plt.imshow(grid.permute(1, 2, 0))
# plt.show()

print(np.unique(train_loader.dataset.dataset.targets, return_counts=True))


evae = EVAE(image_size=image_size,
            image_channel=image_channel,
            std_channel=std_channel,
            latent_dim=latent_dim,
            num_class=num_class,
            e_norm=None,
            d_norm="bn").to(device)
evae.apply(initialize_weights)

g = Generator(image_size=image_size,
              image_channel=image_channel,
              std_channel=std_channel,
              latent_dim=latent_dim,
              num_class=num_class,
              norm='bn').to(device)
d = Discriminator(image_size=image_size,
                  image_channel=image_channel,
                  std_channel=std_channel,
                  latent_dim=latent_dim,
                  num_class=num_class,
                  norm=None).to(device)


# for i in dict(eae.state_dict()):
#     print(i)
#
# for i in dict(g.named_parameters()):
#     print(i)
#
# for i in dict(d.named_parameters()):
#     print(i)


def loss_function(recon_x, x, mu, log_var):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    # KLD = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

    return BCE + KLD

# eae optimizer and loss function
evae_optimizer = Adam(evae.parameters(),
                     lr=learning_rate,
                     betas=(beta1, beta2))
mse_loss = nn.MSELoss()


for epoch in range(eae_num_epoch):
    train_loss = 0
    for idx, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)

        decode, mu, log_var = evae(image, label)
        loss = mse_loss(decode, image)
        loss = loss_function(decode, image, mu, log_var)
        evae_optimizer.zero_grad()
        loss.backward()
        evae_optimizer.step()

        train_loss += loss.item()

    print(f"Epoch: {epoch + 1}/{eae_num_epoch} ({idx}), Loss: {train_loss/len(train_loader)}")

with torch.no_grad():
    evae.eval()
    fixed_vector_output, mu, log_var = evae(fixed_image[target_idx].to(device), fixed_label[target_idx].to(device))
    grid = make_grid(fixed_vector_output.detach().cpu(), nrow=10, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()

# Save AE
torch.save(evae.state_dict(), WEIGHT_PATH + f"evae_{eae_num_epoch}.pth")


# init G model
g.load_state_dict(evae.state_dict(), strict=False)
d.load_state_dict(evae.state_dict(), strict=False)



d_optimizer = Adam(d.parameters(),
                 lr=learning_rate,
                 weight_decay=1e-5,
                 betas=(beta1, beta2))
g_optimizer = Adam(g.parameters(),
                 lr=learning_rate,
                 weight_decay=1e-5,
                 betas=(beta1, beta2))

def d_loss_function(real_logit, fake_logit, wrong_logit):
    real_loss = F.binary_cross_entropy_with_logits(real_logit, torch.ones_like(real_logit).to(device))
    fake_loss = F.binary_cross_entropy_with_logits(fake_logit, torch.zeros_like(fake_logit).to(device))
    wrong_loss = F.binary_cross_entropy_with_logits(wrong_logit, torch.zeros_like(wrong_logit).to(device))
    return real_loss + fake_loss + wrong_loss

def g_loss_function(gen_img_logit):
    fake_loss = F.binary_cross_entropy_with_logits(gen_img_logit, torch.ones_like(gen_img_logit).to(device))
    return fake_loss

def compute_gradient_penalty(D, real_samples, fake_samples, real_labels):
    # real_samples = real_samples.reshape(real_samples.size(0), 1, 28, 28).to(device)
    # fake_samples = fake_samples.reshape(fake_samples.size(0), 1, 28, 28).to(device)

    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples.data + ((1 - alpha) * fake_samples.data)).requires_grad_(True)
    d_interpolates = D(interpolates, real_labels)

    weights = torch.ones(d_interpolates.size()).to(device)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=weights,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradients2L2norm = torch.sqrt(torch.sum(gradients ** 2, dim=1))
    gradient_penalty = torch.mean(( gradients2L2norm - 1 ) ** 2)
    return gradient_penalty

def plt_img(epoch):
    with torch.no_grad():
        g.eval()
        fixed_vector_output = g(fixed_noise, fixed_noise_label)
        grid = make_grid(fixed_vector_output.detach().cpu(), nrow=10, normalize=True)
        plt.axis('off')
        plt.imshow(grid.permute(1, 2, 0))
        plt.savefig(IMG_PATH + 'generated_plot_%d.png' % epoch)
        plt.show()
    return


for epoch in range(gan_num_epoch):
    train_loss = 0
    for idx, (real_image, real_label) in enumerate(train_loader):
        real_image = real_image.to(device)
        real_label = real_label.to(device)

        _batch = real_image.size(0)

        for i in range(train_ratio):
            noise = torch.randn(_batch, latent_dim).to(device)
            wrong_label = (torch.rand(_batch, 1) * 10).long().to(device)
            fake_label = (torch.rand(_batch, 1) * 10).long().to(device)

            fake_image = g(noise, fake_label)
            fake_logit = d(fake_image.detach(), fake_label)
            real_logit = d(real_image, real_label)
            wrong_logit = d(real_image, wrong_label)

            d_cost = d_loss_function(real_logit=real_logit, fake_logit=fake_logit, wrong_logit=wrong_logit)
            gp = compute_gradient_penalty(D=d, real_samples=real_image, fake_samples=fake_image, real_labels=real_label)
            d_loss = d_cost + gp_weight * gp

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

        noise = torch.randn(_batch, latent_dim).to(device)
        fake_label = (torch.rand(_batch, 1) * 10).long().to(device)

        generated_image = g(noise, fake_label)
        gen_img_logit = d(generated_image, fake_label)
        g_loss = g_loss_function(gen_img_logit)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        print(f"Epoch: {epoch+1}, index: {idx}/{len(train_loader)}, D_loss: {d_loss}, G_loss: {g_loss}")
    plt_img(epoch)

    torch.save(g.state_dict(), WEIGHT_PATH + f"g_{epoch}.pth")
    torch.save(d.state_dict(), WEIGHT_PATH + f"d_{epoch}.pth")

# save gif
import imageio
ims = []
for i in range(gan_num_epoch):
    fname = 'generated_plot_%d.png' % i
    dir = IMG_PATH
    if fname in os.listdir(dir):
        print('loading png...', i)
        im = imageio.imread(dir + fname, 'png')
        ims.append(im)
print('saving as gif...')
imageio.mimsave(dir + 'training_demo.gif', ims, fps=3)


    #
    #     fake_image = G(noise)
    #
    #     ################################################# Train Generator
    #     d_optimizer.zero_grad()
    #     d_output_real = D(image)
    #     d_loss_real = bce_loss(d_output_real, real_label)
    #
    #     d_output_fake = D(fake_image.detach())
    #     d_loss_fake = bce_loss(d_output_fake, fake_label)
    #
    #     d_loss = 0.5 * (d_loss_real + d_loss_fake)
    #     d_loss.backward()
    #     d_optimizer.step()
    #
    #
    #     ################################################# Train Discriminator
    #     g_optimizer.zero_grad()
    #     g_output = D(fake_image)
    #     g_loss = bce_loss(g_output, real_label)
    #
    #     g_loss.backward()
    #     g_optimizer.step()
    #
    #
    # print(f"Epoch: {epoch+1}/{num_epoch}, "
    #       f"D Score: {d_loss.item()/len(train_loader):.4f}"
    #       f"G Score: {g_loss.item()/len(train_loader):.4f}")
    #
    # with torch.no_grad():
    #     G.eval()
    #     fixed_vector_output = G(fixed_noise)
    #     grid = make_grid(fixed_vector_output.detach().cpu(), nrow=10, normalize=True)
    #     plt.imshow(grid.permute(1,2,0))
    #     plt.show()
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



    

