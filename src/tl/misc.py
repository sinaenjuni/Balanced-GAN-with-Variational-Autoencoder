import torch.nn as nn
import tl.opt as opt

class make_empty_object(object):
    pass




MODULE = make_empty_object()

# MODULE.g_deconv = opt.deconv2d
# MODULE.g_bn = opt.batchnorm_2d
# MODULE.g_act_fn = nn.LeakyReLU(negative_slope=0.2, inplace=True)
# MODULE.g_linear = opt.linear

MODULE.g_deconv = opt.sndeconv2d
MODULE.g_bn = opt.batchnorm_2d
MODULE.g_act_fn = nn.LeakyReLU(negative_slope=0.2, inplace=True)
MODULE.g_linear = opt.snlinear
