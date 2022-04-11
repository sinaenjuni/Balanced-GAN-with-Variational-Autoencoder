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

