import numpy as np
from scipy.linalg import sqrtm
from metric.inception_net import InceptionV3



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


def stack_real_data_features(eval_model, data_loader, resizer, device):
    eval_model.to(device)
    real_image_features = []
    labels = []

    for idx, (real_image, label) in enumerate(data_loader):
        print(idx)
        real_image = real_image.to(device)
        label = label.to(device)

        real_image = resizer(real_image)

        if real_image.size(1) == 1:
            real_image = real_image.repeat(1,3,1,1)

        real_image_feature, real_image_logit = eval_model(real_image)
        real_image_features.append(real_image_feature.detach().cpu())
        labels.append(label.detach().cpu())

    real_image_features = torch.cat(real_image_features, 0)
    labels = torch.cat(labels, 0)

    return real_image_features, labels



def stack_gen_data_features(eval_model, gen_model, latent_dim, sample_size, num_class, resizer, device):
    gen_image_features = []
    labels = []

    for cls in range(num_class):
        for _ in range(sample_size//10):
            noise = torch.randn(sample_size//100, latent_dim).to(device)
            label = (torch.ones(sample_size//100, dtype=torch.long) * cls).to(device)

            gen_image = gen_model(noise, label)
            gen_image = resizer(gen_image)

            if gen_image.size(1) == 1:
                gen_image = gen_image.repeat(1,3,1,1)

            gen_image_feature, gen_image_logit = eval_model(gen_image)
            gen_image_features.append(gen_image_feature.detach().cpu())
            labels.append([cls] *10)

    gen_image_features = torch.cat(gen_image_features, 0)
    labels = np.array(labels).ravel()

    return gen_image_features, labels


if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader
    from torchvision.transforms import *
    from datasets.sampler import BalancedSampler
    from datasets.imbalance_fashion_mnist import Imbalanced_FashionMNIST
    from datasets.imbalance_cifar import Imbalanced_CIFAR10
    from models.gan.embedded_generator import Generator

    latent_dim = 128
    batch_size = 64
    imb_factor = 0.01
    num_class = 10
    sample_size = 1000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuda0 = torch.device("cuda:0")
    cuda1 = torch.device("cuda:1")
    # device = torch.device('cpu')
    resizer = Resize(299)

    transforms = Compose([
        Resize(64),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = Imbalanced_CIFAR10(root='~/data/',
                           train=True,
                           imb_factor=imb_factor,
                           download=True,
                           transform=transforms)

    # test_dataset = Imbalanced_CIFAR10(root='~/data/',
    #                        train=False,
    #                        imb_factor=imb_factor,
    #                        download=True,
    #                        transform=transforms)


    # sampler = BalancedSampler(train_dataset, retain_epoch_size=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    # fixed_image, fixed_label = iter(train_loader).__next__()


    eval_model = InceptionV3(resize_input=False, normalize_input=False).to(device).eval()
    gen_model = Generator(image_size=64, image_channel=3, std_channel=64, latent_dim=128, num_class=10, norm='bn').to(device).eval()
    # gen_model.load_state_dict(torch.load('/home/sin/git/ae/src/weights/eae/fashion_mnist/g_30.pth'))
    # gen_model.load_state_dict(torch.load('/home/sin/git/ae/src/weights/eae/cifar10/g_99.pth'))
    gen_model.load_state_dict(torch.load('/home/sin/git/ae/src/weights/evae/cifar10/g_99.pth'))

    real_image_features, real_labels = stack_real_data_features(eval_model, train_loader, resizer, device)
    gen_image_features, gen_labels = stack_gen_data_features(eval_model, gen_model, latent_dim, sample_size, num_class, resizer, device)


    for cls in range(num_class):
        target_image_features = real_image_features[real_labels == cls]
        target_gen_features = gen_image_features[gen_labels == cls]
        fid = calculate_fid(target_image_features.numpy(), target_gen_features.numpy())
        print('>>FID(%d): %.3f' % (cls, fid))

    fid = calculate_fid(real_image_features.numpy(), gen_image_features.numpy())

    print('>>FID(all): %.3f' % (fid))
    print('-'*50)