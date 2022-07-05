import os
import torch
from models.gan.embedded_generator import Generator
# from torchvision.utils import save_image
from PIL import Image


image_size = 64
image_channel = 3
std_channel = 64
latent_dim = 128
num_classes = 10
device = 'cuda'

save_path = '/home/dblab/sin/save_files/gen/our/cifar10/'

# fixed_noise = torch.randn(10000, latent_dim).to(device)
# fixed_noise_label = torch.tensor([[i//1000] for i in range(10000)]).to(device)

g = Generator(image_size=image_size,
              image_channel=image_channel,
              std_channel=std_channel,
              latent_dim=latent_dim,
              num_class=num_classes,
              norm='bn').to(device)

state = torch.load("/home/dblab/sin/git/VAE-GAN/src/save_files/evae/ebgan_cifar10/weights/g_99.pth")
g.load_state_dict(state)

for cls_ in range(num_classes):
    gen_img_list = []

    latent = torch.randn(1000, latent_dim).to(device)
    label = torch.ones(1000).to(torch.int).to(device) * cls_
    gen_img = g(latent, label)
    gen_img = gen_img.detach()
    gen_img_list.append(gen_img)
    gen_img_list = torch.cat(gen_img_list)

    for idx, gen_img in enumerate(gen_img_list):
        path_ = os.path.join(save_path, str(cls_))
        if not os.path.exists(path_):
            os.makedirs(path_)
        gen_img = ((gen_img * 0.5 + 0.5)*255.0).to(torch.uint8).permute(1,2,0).cpu().numpy()
        gen_img = Image.fromarray(gen_img)
        # print(gen_img.min(), gen_img.max())
        # save_image(gen_img, path_ + f"/{idx + (1000 * cls_)}.png")
        gen_img.save(path_ + f"/{idx + (1000 * cls_)}.png")

