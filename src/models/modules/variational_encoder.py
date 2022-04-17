import torch
import torch.nn as nn
from models.modules.encoder_module import Encoder_module

class VariationalEncoder(Encoder_module):
    def __init__(self, image_size, image_channel, std_channel, latent_dim):
        super(VariationalEncoder, self).__init__(image_size=image_size, image_channel=image_channel, std_channel=std_channel, latent_dim=latent_dim)

        self.mu      = nn.Linear(self.image_size * self.image_size * std_channel * 4, latent_dim)
        self.log_var = nn.Linear(self.image_size * self.image_size * std_channel * 4, latent_dim)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.feature(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return self.sampling(mu, log_var), mu, log_var

if __name__ == "__main__":
    def initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, std=0.02)

    VE = VariationalEncoder(image_size=64, image_channel=3, std_channel=64, latent_dim=128)
    VE.apply(initialize_weights)

    inputs = torch.randn((128, 3, 64, 64))
    encode, mu, log_var = VE(inputs)
    print(encode.size())
    print(mu.size())
    print(log_var.size())


    for i in VE.named_parameters():
        print(i[0])
