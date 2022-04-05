import torch
import torch.nn as nn


class Encoder(nn.Module):
    def getLayer(self, num_input, num_output, kernel_size, stride, padding):
        return nn.Sequential(nn.Conv2d(in_channels=num_input,
                                       out_channels=num_output,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding),
                             nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def __init__(self, image_size, image_channel, std_channel, latent_dim):
        super(Encoder, self).__init__()

        image_size = image_size // 2 ** 4

        self.layer1 = self.getLayer(image_channel, std_channel,   kernel_size=4, stride=2, padding=1)
        self.layer2 = self.getLayer(std_channel,   std_channel*2, kernel_size=4, stride=2, padding=1)
        self.layer3 = self.getLayer(std_channel*2, std_channel*2, kernel_size=4, stride=2, padding=1)
        self.layer4 = self.getLayer(std_channel*2, std_channel*4, kernel_size=4, stride=2, padding=1)
        # self.layer5 = nn.Sequential(nn.Linear(image_size * image_size * std_channel*4, latent_dim),
        #                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.layer5 = nn.Linear(image_size * image_size * std_channel * 4, latent_dim)
        self.layer5_1 = nn.Linear(image_size * image_size * std_channel * 4, latent_dim)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(x, start_dim=1)
        # x = self.layer5(x)
        return self.sampling(self.layer5(x), self.layer5_1(x)), self.layer5(x), self.layer5_1(x)


if __name__ == "__main__":
    def initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, std=0.02)

    E = Encoder(image_size=64, image_channel=3, std_channel=64, latent_dim=128)
    E.apply(initialize_weights)

    inputs = torch.randn((128, 3, 64, 64))
    encode, mu, log_var = E(inputs)
    print(encode.size())
    print(mu.size())
    print(log_var.size())

