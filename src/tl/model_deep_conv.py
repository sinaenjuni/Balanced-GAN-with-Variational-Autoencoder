import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import tl.misc as misc

class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, MODULES):
        super(GenBlock, self).__init__()

        self.deconv = MODULES.g_deconv(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1)
        self.bn = MODULES.g_bn(out_channels)
        self.activation = MODULES.g_act_fn

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class Generator_(nn.Module):
    def __init__(self, latent_dim, MODULES):
        super(Generator_, self).__init__()
        self.in_dims = [128, 128, 64]
        self.out_dims = [128, 128, 3]


        self.in_layer = MODULES.g_linear(in_features=latent_dim,
                                       out_feature= self.in_dims[0] * (4 * 4),
                                       bias=True)

        self.blocks = []
        for index in range(len(self.in_dims)):
            self.blocks += [[
                GenBlock(in_channels=self.in_dims[index],
                         out_channels=self.out_dims[index],
                         MODULES=MODULES)
            ]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.out_layer = MODULES.g_deconv(in_channels=self.in_dims[-1],
                                    out_channels=self.out_dims[-1],
                                    kernel_size=4,
                                    stride=2,
                                    padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(-1, self.in_dims[0], 4, 4)

        for blocklist in self.blocks:
            for block in blocklist:
                x = block(x)

        x = self.tanh(x)
        # x = torch.sigmoid_(x)
        return x


if __name__ == "__main__":
    def initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, std=0.02)
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data, std=0.02)

    gen_block = GenBlock(256, 128, misc.MODULE)


    G = Generator_(128, misc.MODULE)

    D.apply(initialize_weights)

    inputs = torch.randn((1, 128))
    outputs = D(inputs)
    print(outputs.size())

    for i in D.named_parameters():
        print(i[0])
