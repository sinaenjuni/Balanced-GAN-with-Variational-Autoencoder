import torch
import torch.nn as nn
from models.modules.encoder_module import Encoder_module

class Discriminator(Encoder_module):
    def __init__(self, image_size, image_channel, std_channel, latent_dim):
        super(Discriminator, self).__init__(image_size, image_channel, std_channel, latent_dim)

        self.adv = nn.Linear(self.image_size * self.image_size * self.channels[-1], 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.feature(x)
        # x = torch.flatten(x, start_dim=1)
        # x = self.layer5(x)
        x = self.adv(x)
        x = torch.sigmoid_(x)
        return x


if __name__ == "__main__":
    def initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, std=0.02)

    E = Discriminator(image_size=64, image_channel=3, std_channel=64, latent_dim=128)
    E.apply(initialize_weights)

    inputs = torch.randn((128, 3, 64, 64))
    outputs = E(inputs)
    print(outputs.size())

