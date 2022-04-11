import torch.nn as nn


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, std=0.02)
        # nn.init.normal_(m.bias.data, std=0.02)
    if isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data, std=0.02)
        # nn.init.normal_(m.bias.data, std=0.02)
