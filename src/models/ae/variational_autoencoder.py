import torch
import torch.nn as nn
from models.modules.variational_encoder import VariationalEncoder
from models.modules.decoder_module import Decoder_module


class VAE(nn.Module):
    def __init__(self, image_size, image_channel, std_channel, latent_dim):
        super(VAE, self).__init__()
        self.vencoder = VariationalEncoder(image_size=image_size,
                                      image_channel=image_channel,
                                      std_channel=std_channel,
                                      latent_dim=latent_dim)
        self.decoder = Decoder_module(image_size=image_size,
                                      image_channel=image_channel,
                                      std_channel=std_channel,
                                      latent_dim=latent_dim)


    def forward(self, x):
        vencode, mu, log_var = self.vencoder(x)
        decode = self.decoder(vencode)

        return decode, mu, log_var


if __name__ == "__main__":

    def initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, std=0.02)

    E = VariationalEncoder(image_size=64, image_channel=3, std_channel=64, latent_dim=128)
    E.apply(initialize_weights)

    inputs = torch.randn((128, 3, 64, 64))
    encode, mu, log_var = E(inputs)

    print(encode.size())
    print(mu.size())
    print(log_var.size())


    for i in E.named_parameters():
        print(i[0])