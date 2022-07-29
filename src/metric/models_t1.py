import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import sr.opt as opt


class Decoder(nn.Module):
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.02)
                # if m.bias is not None:
                #     nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                # nn.init.constant_(m.bias, 0)

    def __init__(self, img_dim, latent_dim, num_classes):
        super(Decoder, self).__init__()
        self.dims = [256, 128, 128, 64, img_dim]

        self.linear0 = nn.Sequential(opt.snlinear(in_features=latent_dim, out_features= self.dims[0] * (4 * 4)),
                                     opt.ConditionalBatchNorm2d(in_features=num_classes, out_features=self.dims[0] * (4 * 4)),
                                     nn.ReLU(inplace=True))

        self.deconv0 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                     opt.snconv2d(in_channels=self.dims[0], out_channels=self.dims[1], kernel_size=3, stride=1, padding=1),
                                     opt.ConditionalBatchNorm2d(in_features=num_classes, out_features=self.dims[1]),
                                     nn.ReLU(inplace=True))

        self.deconv1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                     opt.snconv2d(in_channels=self.dims[1], out_channels=self.dims[2], kernel_size=3, stride=1, padding=1),
                                     opt.ConditionalBatchNorm2d(in_features=num_classes, out_features=self.dims[2]),
                                     nn.ReLU(inplace=True))

        self.deconv2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                     opt.snconv2d(in_channels=self.dims[2], out_channels=self.dims[3], kernel_size=3, stride=1, padding=1),
                                     opt.ConditionalBatchNorm2d(in_features=num_classes, out_features=self.dims[3]),
                                     nn.ReLU(inplace=True))

        self.deconv3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                     opt.snconv2d(in_channels=self.dims[3], out_channels=self.dims[4], kernel_size=3, stride=1, padding=1))

        self.tanh = nn.Tanh()

        self.initialize_weights()

    def forward(self, x, one_hot):
        x = self.linear0(x)
        x = x.view(-1, self.dims[0], 4, 4)
        x = self.deconv0(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.tanh(x)
        return x



class Encoder(nn.Module):
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.02)
                # if m.bias is not None:
                #     nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                # nn.init.constant_(m.bias, 0)

    def __init__(self, img_dim, latent_dim):
        super(Encoder, self).__init__()
        self.dims = [64, 128, 128, 256]

        self.conv0 = nn.Sequential(spectral_norm(nn.Conv2d(in_channels=img_dim, out_channels=self.dims[0], kernel_size=3, stride=2, padding=1)),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.conv1 = nn.Sequential(spectral_norm(nn.Conv2d(in_channels=self.dims[0], out_channels=self.dims[1], kernel_size=3, stride=2, padding=1)),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.conv2 = nn.Sequential(spectral_norm(nn.Conv2d(in_channels=self.dims[1], out_channels=self.dims[2], kernel_size=3, stride=2, padding=1)),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.conv3 = nn.Sequential(spectral_norm(nn.Conv2d(in_channels=self.dims[2], out_channels=self.dims[3], kernel_size=3, stride=2, padding=1)),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.linear0 = nn.Sequential(nn.Flatten(1),
                                     nn.Linear(in_features=self.dims[3] * (4 * 4), out_features=latent_dim),
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True)
                                     )


    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.linear0(x)
        return x

    def getFeatures(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class Embedding_labeled_latent(nn.Module):
    def __init__(self, latent_dim, num_class):
        super(Embedding_labeled_latent, self).__init__()

        self.embedding = nn.Sequential(nn.Embedding(num_embeddings=num_class, embedding_dim=latent_dim),
                                      nn.Flatten())

    def forward(self, z, label):
        le = z * self.embedding(label)
        return le


if __name__ == '__main__':
    batch_size = 100
    num_classes = 10
    img = torch.randn(batch_size, 3, 64, 64)
    z = torch.randn(batch_size, 128)
    label = torch.randint(0, 10, (batch_size,))
    one_hot = F.one_hot(label).to(torch.float)

    encoder = Encoder(3, 128, num_classes)
    decoder = Decoder(3, 128)

    output_decoder = decoder(z, one_hot)
    print(output_decoder.size())



