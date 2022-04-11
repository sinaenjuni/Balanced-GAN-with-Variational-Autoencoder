import torch
import torch.nn as nn
from models.modules.encoder_module import Encoder_module

class Discriminator(nn.Module):
    def __init__(self, image_size, image_channel, std_channel, latent_dim, num_class):
        super(Discriminator, self).__init__()
        self.encoder = Encoder_module(image_size=image_size, image_channel=image_channel, std_channel=std_channel, latent_dim=latent_dim)

        self.em = nn.Sequential(nn.Embedding(num_class, 512),
                                nn.Flatten(start_dim=1),
                                nn.Linear(512, self.encoder.image_size*self.encoder.image_size*self.encoder.channels[-1]),
                                nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.adv = nn.Sequential(nn.Linear(self.encoder.image_size*self.encoder.image_size*self.encoder.channels[-1], 512),
                                 nn.Linear(512, 1))

    def forward(self, x, label):
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.feature(x)

        le = self.em(label)
        x_y = torch.multiply(x, le)
        x_y = self.adv(x_y)
        return x_y


if __name__ == "__main__":
    def initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, std=0.02)

    E = Discriminator(image_size=64, image_channel=3, std_channel=64, latent_dim=128, num_class=10)
    E.apply(initialize_weights)


    inputs = torch.randn((128, 3, 64, 64))
    labels = (torch.rand(128)*10).long()

    outputs = E(inputs, labels)
    print(outputs.size())

