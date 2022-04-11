import torch
import torch.nn as nn
from models.modules.decoder_module import Decoder_module
from models.modules.embedding_module import Embedding_module

class Generator(nn.Module):
    def __init__(self, image_size, image_channel, std_channel, latent_dim, num_class, norm):
        super(Generator, self).__init__()


        self.em = Embedding_module(latent_dim=latent_dim, num_class=num_class)
        self.decoder = Decoder_module(image_size=image_size,
                                      image_channel=image_channel,
                                      std_channel=std_channel,
                                      latent_dim=latent_dim,
                                      norm=norm)


    def forward(self, x, label):
        x = self.em(x, label)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    def initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, std=0.02)


    G = Generator(image_size=64, image_channel=3, std_channel=64, latent_dim=128, num_class=10, norm='bn')
    G.apply(initialize_weights)

    noise = torch.randn(100, 128)
    labels = (torch.rand(100,)*10).long()
    outputs = G(noise, labels)
    print(outputs.size())

    for i in dict(G.named_parameters()):
        print(i)
