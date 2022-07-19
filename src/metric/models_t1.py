import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class Decoder_block(nn.Module):
    def __init__(self, in_featrues, out_featrues):
        super(Decoder_block, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_featrues = in_featrues)
        self.bn2 = nn.BatchNorm2d(in_featrues = out_featrues)

        self.act_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv0 = nn.Conv2d(in_channels=in_featrues, out_featrues=out_featrues, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(in_channels=in_featrues, out_featrues=out_featrues, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_featrues, out_featrues=out_featrues, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        ori = x
        x = self.bn1(x)
        x = self.act_fn(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.act_fn(x)
        x = self.conv2(x)

        ori = F.interpolate(x, scale_factor=2 , mode='nearest')
        ori = self.conv2(ori)
        out = x + ori
        return out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        g_in_dims_collection = {
            "32": [g_conv_dim * 4, g_conv_dim * 4, g_conv_dim * 4],
            "64": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
            "128": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
            "256": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
            "512": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim]
        }

        g_out_dims_collection = {
            "32": [g_conv_dim * 4, g_conv_dim * 4, g_conv_dim * 4],
            "64": [g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
            "128": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
            "256": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
            "512": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim, g_conv_dim]
        }

    def forward(self, x):
        



# class Decoder(nn.Module):
#     def initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight, std=0.02)
#                 # if m.bias is not None:
#                 #     nn.init.constant_(m.bias, 0)
#             # elif isinstance(m, nn.BatchNorm2d):
#             #     nn.init.constant_(m.weight, 1)
#             #     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, std=0.02)
#                 # nn.init.constant_(m.bias, 0)
#
#
#     def __init__(self, img_dim, latent_dim):
#         super(Decoder, self).__init__()
#         self.dims = [256, 128, 128, 64, img_dim]
#
#         self.linear0 = nn.Sequential(nn.Linear(in_features=latent_dim, out_features= self.dims[0] * (4 * 4)),
#                                      nn.LeakyReLU(negative_slope=0.2, inplace=True))
#
#         self.deconv0 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), n
#                                      spectral_norm(nn.Conv2d(in_channels=self.dims[0], out_channels=self.dims[1],
#                                                         kernel_size=3, stride=1, padding=1)),
#                                      nn.BatchNorm2d(self.dims[1]),
#                                      nn.LeakyReLU(negative_slope=0.2, inplace=True))
#
#         self.deconv1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
#                                      spectral_norm(nn.Conv2d(in_channels=self.dims[1], out_channels=self.dims[2],
#                                                         kernel_size=3, stride=1, padding=1)),
#                                      nn.BatchNorm2d(self.dims[2]),
#                                      nn.LeakyReLU(negative_slope=0.2, inplace=True))
#
#         self.deconv2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
#                                      spectral_norm(nn.Conv2d(in_channels=self.dims[2], out_channels=self.dims[3],
#                                                         kernel_size=3, stride=1, padding=1)),
#                                      nn.BatchNorm2d(self.dims[3]),
#                                      nn.LeakyReLU(negative_slope=0.2, inplace=True))
#
#         self.deconv3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
#                                      spectral_norm(nn.Conv2d(in_channels=self.dims[3], out_channels=self.dims[4],
#                                                              kernel_size=3, stride=1, padding=1)))
#
#         # self.deconv0 = self.getLayer(self.dims[0], self.dims[1], kernel_size=4, stride=2, padding=1)   # 4*4*128
#         # self.deconv1 = self.getLayer(self.dims[1], self.dims[2], kernel_size=4, stride=2, padding=1)   # 8*8*128
#         # self.deconv2 = self.getLayer(self.[2], self.dims[3], kernel_size=4, stride=2, padding=1)   # 16*16*64
#         # self.deconv3 = nn.Sequential(nn.ConvTranspose2d(std_channel*1, image_channel, kernel_size=4, stride=2, padding=1))   # 32*32*3
#
#         self.initialize_weights()
#
#     def forward(self, x):
#         x = self.linear0(x)
#         x = x.view(-1, self.dims[0], 4, 4)
#         x = self.deconv0(x)
#         x = self.deconv1(x)
#         x = self.deconv2(x)
#         x = self.deconv3(x)
#         x = torch.tanh_(x)
#         return x
#
#
#
# class Encoder(nn.Module):
#     # def getLayer(self, num_input, num_output, kernel_size, stride, padding, norm, is_flatten):
#     #     if norm == "sp":
#     #         conv = spectral_norm(nn.Conv2d(in_channels=num_input,
#     #                                    out_channels=num_output,
#     #                                    kernel_size=kernel_size,
#     #                                    stride=stride,
#     #                                    padding=padding))
#     #     else:
#     #         conv = nn.Conv2d(in_channels=num_input,
#     #                                        out_channels=num_output,
#     #                                        kernel_size=kernel_size,
#     #                                        stride=stride,
#     #                                        padding=padding)
#     #
#     #     lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#     #     flatten = nn.Flatten(start_dim=1)
#     #
#     #     if is_flatten:
#     #         layer = nn.Sequential(conv, lrelu, flatten)
#     #     else:
#     #         layer = nn.Sequential(conv, lrelu)
#     #
#     #     return layer
#     #
#     # def sampling(self, mu, log_var):
#     #     std = torch.exp(0.5 * log_var)
#     #     eps = torch.randn_like(std)
#     #     return eps.mul(std).add_(mu)  # return z sample
#
#     def initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight, std=0.02)
#                 # if m.bias is not None:
#                 #     nn.init.constant_(m.bias, 0)
#             # elif isinstance(m, nn.BatchNorm2d):
#             #     nn.init.constant_(m.weight, 1)
#             #     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, std=0.02)
#                 # nn.init.constant_(m.bias, 0)
#
#     def __init__(self, img_dim, latent_dim):
#         super(Encoder, self).__init__()
#         self.dims = [64, 128, 128, 256]
#
#         self.conv0 = nn.Sequential(spectral_norm(nn.Conv2d(in_channels=img_dim, out_channels=self.dims[0], kernel_size=3, stride=2, padding=1)),
#                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
#
#         self.conv1 = nn.Sequential(spectral_norm(nn.Conv2d(in_channels=self.dims[0], out_channels=self.dims[1], kernel_size=3, stride=2, padding=1)),
#                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
#
#         self.conv2 = nn.Sequential(spectral_norm(nn.Conv2d(in_channels=self.dims[1], out_channels=self.dims[2], kernel_size=3, stride=2, padding=1)),
#                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
#
#         self.conv3 = nn.Sequential(spectral_norm(nn.Conv2d(in_channels=self.dims[2], out_channels=self.dims[3], kernel_size=3, stride=2, padding=1)),
#                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
#
#         self.linear0 = nn.Sequential(nn.Flatten(1),
#                                      nn.Linear(in_features=self.dims[3] * (4 * 4), out_features=latent_dim),
#                                      nn.LeakyReLU(negative_slope=0.2, inplace=True)
#                                      )
#
#         # self.image_size = image_size // 2 ** 4
#         # self.channels = [std_channel, std_channel*2, std_channel*4]
#         #
#         # self.layer1 = self.getLayer(image_channel,    self.channels[0], kernel_size=4, stride=2, padding=1, is_flatten=False, norm=norm)
#         # self.layer2 = self.getLayer(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1, is_flatten=False, norm=norm)
#         # self.layer3 = self.getLayer(self.channels[1], self.channels[1], kernel_size=4, stride=2, padding=1, is_flatten=False, norm=norm)
#         # self.feature = self.getLayer(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1, is_flatten=True, norm=norm)
#
#         # if norm == "sp":
#         #     self.fc_lrelu = nn.Sequential(spectral_norm(nn.Linear(self.image_size * self.image_size * self.channels[-1], latent_dim)),
#         #                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))
#         # else:
#         #     self.fc_lrelu = nn.Sequential(nn.Linear(self.image_size * self.image_size * self.channels[-1], latent_dim),
#         #                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))
#
#     def forward(self, x):
#         x = self.conv0(x)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.linear0(x)
#         return x
#
#     def getFeatures(self, x):
#         x = self.conv0(x)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         return x
#
# class Embedding_labeled_latent(nn.Module):
#     def __init__(self, latent_dim, num_class):
#         super(Embedding_labeled_latent, self).__init__()
#
#         self.embedding = nn.Sequential(nn.Embedding(num_embeddings=num_class, embedding_dim=latent_dim),
#                                       nn.Flatten())
#
#     def forward(self, z, label):
#         le = z * self.embedding(label)
#         return le


if __name__ == '__main__':


    encoder = Encoder(3, 128)
    decoder = Decoder(3, 128)

    z = torch.randn(100, 128)
    output_decoder = decoder(z)
    print(output_decoder.size())


