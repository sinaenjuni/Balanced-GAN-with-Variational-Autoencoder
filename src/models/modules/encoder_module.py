import torch
import torch.nn as nn


class Encoder_module(nn.Module):
    def getLayer(self, num_input, num_output, kernel_size, stride, padding, is_flatten):
        conv = nn.Conv2d(in_channels=num_input,
                                       out_channels=num_output,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding)
        lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        flatten = nn.Flatten(start_dim=1)

        if is_flatten:
            layer = nn.Sequential(conv, lrelu, flatten)
        else:
            layer = nn.Sequential(conv, lrelu)

        return layer

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def __init__(self, image_size, image_channel, std_channel, latent_dim):
        super(Encoder_module, self).__init__()

        self.image_size = image_size // 2 ** 4
        self.channels = [std_channel, std_channel*2, std_channel*4]

        self.layer1 = self.getLayer(image_channel,    self.channels[0], kernel_size=4, stride=2, padding=1, is_flatten=False)
        self.layer2 = self.getLayer(self.channels[0], self.channels[1], kernel_size=4, stride=2, padding=1, is_flatten=False)
        self.layer3 = self.getLayer(self.channels[1], self.channels[1], kernel_size=4, stride=2, padding=1, is_flatten=False)
        self.feature = self.getLayer(self.channels[1], self.channels[2], kernel_size=4, stride=2, padding=1, is_flatten=True)

        self.fc_lrelu = nn.Sequential(nn.Linear(self.image_size * self.image_size * self.channels[-1], latent_dim),
                                          nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.feature(x)
        x = self.fc_lrelu(x)
        return x



if __name__ == "__main__":
    def initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, std=0.02)

    E = Encoder_module(image_size=64, image_channel=3, std_channel=64, latent_dim=128)
    E.apply(initialize_weights)

    inputs = torch.randn((128, 3, 64, 64))
    outputs = E(inputs)
    print(outputs.size())

    for i in E.named_parameters():
        print(i[0])