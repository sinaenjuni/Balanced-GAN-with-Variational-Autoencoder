import torch
import torch.nn as nn
from models.modules.encoder_module import Encoder_module
from models.modules.decoder_module import Decoder_module
from models.modules.embedding_module import Embedding_module

class EAE(nn.Module):
    def __init__(self, image_size, image_channel, std_channel, latent_dim, num_class, norm):
        super(EAE, self).__init__()
        self.encoder = Encoder_module(image_size=image_size,
                                      image_channel=image_channel,
                                      std_channel=std_channel,
                                      latent_dim=latent_dim)
        self.em = Embedding_module(latent_dim=latent_dim, num_class=num_class)
        self.decoder = Decoder_module(image_size=image_size,
                                      image_channel=image_channel,
                                      std_channel=std_channel,
                                      latent_dim=latent_dim,
                                      norm=norm)
    def forward(self, x, y):
        encode = self.encoder(x)
        embedded_encode = self.em(encode, y)
        decode = self.decoder(embedded_encode)
        return decode


if __name__ == "__main__":
    ae = EAE(image_size=64, image_channel=3, std_channel=64, latent_dim=128, num_class=10, norm='bn')
    for pname in dict(ae.state_dict()):
        print(pname)
    # print(list(ae.encoder.modules()))
    # print(ae.encoder.layer2[0].weight)