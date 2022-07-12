import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import misc as misc
import opt as opt

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
    def __init__(self, latent_dim, img_channels, MODULES):
        super(Generator_, self).__init__()
        self.in_dims = [128, 128, 64, 32]
        self.out_dims = [128, 64, 32, 16]
        self.img_channels = img_channels
        self.g_init = MODULES.g_init

        self.in_layer = MODULES.g_linear(in_features = latent_dim,
                                       out_features = self.in_dims[0] * (4 * 4),
                                       bias = True)

        self.blocks = []
        for in_dims, out_dims in zip(self.in_dims, self.out_dims):
            self.blocks += [[
                GenBlock(in_channels=in_dims,
                         out_channels=out_dims,
                         MODULES=MODULES)
            ]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.out_layer = MODULES.g_deconv(in_channels = self.out_dims[-1],
                                    out_channels = self.img_channels,
                                    kernel_size = 3,
                                    stride = 1,
                                    padding = 1)
        self.tanh = nn.Tanh()

        # print(self.modules())
        opt.init_weights(self.modules, self.g_init)

    def forward(self, x):
        x = self.in_layer(x)
        x = x.view(-1, self.in_dims[0], 4, 4)

        for blocklist in self.blocks:
            for block in blocklist:
                x = block(x)

        x = self.out_layer(x)
        x = self.tanh(x)
        # x = torch.sigmoid_(x)
        return x


class DisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, MODULES):
        super(DisBlock, self).__init__()
        self.d_sn = MODULES.d_sn

        self.conv0 = MODULES.d_conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv1 = MODULES.d_conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

        if not self.d_sn:
            self.bn0 = MODULES.d_bn(out_channels)
            self.bn1 = MODULES.d_bn(out_channels)

        self.activation = MODULES.d_act_fn

    def forward(self, x):
        x = self.conv0(x)
        if not self.d_sn:
            x = self.bn0(x)
        x = self.activation(x)

        x = self.conv1(x)
        if not self.d_sn:
            x = self.bn1(x)
        x = self.activation(x)
        return x


class Discriminator_(nn.Module):
    def __init__(self, img_channels, MODULES):
        super(Discriminator_, self).__init__()
        self.in_dims  = [img_channels, 32,  64]
        self.out_dims = [32,          64, 128]
        self.is_bn = not MODULES.d_sn
        self.d_init = MODULES.d_init


        self.blocks = []
        for index in range(len(self.in_dims)):
            self.blocks += [[
                DisBlock(in_channels=self.in_dims[index], out_channels=self.out_dims[index], MODULES=MODULES)
            ]]
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.activation = MODULES.d_act_fn

        self.conv1 = MODULES.d_conv2d(in_channels=self.out_dims[-1], out_channels=512, kernel_size=3, stride=1, padding=1)

        if self.is_bn:
            self.bn1 = MODULES.d_bn(512)


        self.linear1 = MODULES.d_linear(in_features=512, out_features=1, bias=True)

        opt.init_weights(self.modules, self.d_init)


    def forward(self, x):

        for blocklist in self.blocks:
            for block in blocklist:
                x = block(x)
        x = self.conv1(x)

        if self.is_bn:
            x = self.bn1(x)
        x = self.activation(x)
        x = torch.sum(x, dim=[2, 3])

        adv_output = torch.squeeze(self.linear1(x))
        return adv_output


if __name__ == "__main__":
    # def initialize_weights(m):
    #     if isinstance(m, nn.Conv2d):
    #         nn.init.normal_(m.weight.data, std=0.02)
    #     if isinstance(m, nn.ConvTranspose2d):
    #         nn.init.normal_(m.weight.data, std=0.02)

    # gen_block = GenBlock(256, 128, misc.MODULES)
    G = Generator_(latent_dim=128,
                   img_channels=3,
                   MODULES=misc.MODULES).cuda()

    # G.apply(initialize_weights)

    z = torch.randn((100, 128)).cuda()
    gened_img = G(z)
    print(z.shape, gened_img.shape)


    dis_block = DisBlock(3, 64, misc.MODULES)
    print(dis_block)
    D = Discriminator_(img_channels=3, MODULES=misc.MODULES).cuda()
    real = torch.randn(100, 3, 64, 64).cuda()
    outputs = D(real)
    print(outputs.shape)

    print(D(gened_img))


    # print(outputs.size())
    #
    # for i in G.named_parameters():
    #     print(i[0])
