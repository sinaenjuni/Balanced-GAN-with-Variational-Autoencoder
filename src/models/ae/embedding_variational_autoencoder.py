import torch
import torch.nn as nn
from models.modules.variational_encoder import VariationalEncoder
from models.modules.decoder_module import Decoder_module
from models.modules.embedding_module import Embedding_module


class EVAE(nn.Module):
    def __init__(self, image_size, image_channel, std_channel, latent_dim, num_class, e_norm, d_norm):
        super(EVAE, self).__init__()
        self.encoder = VariationalEncoder(image_size=image_size,
                                      image_channel=image_channel,
                                      std_channel=std_channel,
                                      latent_dim=latent_dim,
                                          norm=e_norm)
        self.em = Embedding_module(latent_dim=latent_dim, num_class=num_class)
        self.decoder = Decoder_module(image_size=image_size,
                                      image_channel=image_channel,
                                      std_channel=std_channel,
                                      latent_dim=latent_dim,
                                      norm=d_norm)


    def forward(self, x, y):
        vencode, mu, log_var = self.encoder(x)
        embedded_encode = self.em(vencode, y)
        decode = self.decoder(embedded_encode)

        return decode, mu, log_var


if __name__ == "__main__":

    def initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, std=0.02)

    evae = EVAE(image_size=64, image_channel=3, std_channel=64, latent_dim=128, num_class=10, e_norm="sp", d_norm="bn")
    evae.apply(initialize_weights)

    # inputs = torch.randn((128, 3, 64, 64))
    # encode, mu, log_var = evae(inputs)
    #
    # print(encode.size())
    # print(mu.size())
    # print(log_var.size())


    for i in evae.named_parameters():
        print(i[0])