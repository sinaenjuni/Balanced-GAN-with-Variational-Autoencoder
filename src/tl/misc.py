import audioop

import torch.nn as nn
import opt as opt

class make_empty_object(object):
    pass




MODULES = make_empty_object()

# MODULE.g_deconv = opt.deconv2d
# MODULE.g_bn = opt.batchnorm_2d
# MODULE.g_act_fn = nn.LeakyReLU(negative_slope=0.2, inplace=True)
# MODULE.g_linear = opt.linear

MODULES.g_deconv = opt.deconv2d
MODULES.g_sn = False
MODULES.g_bn = opt.batchnorm_2d
# MODULE.g_act_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
MODULES.g_linear = opt.linear
MODULES.g_act_fn = nn.ReLU(True)
MODULES.g_init = 'ortho'

MODULES.d_conv2d = opt.conv2d
MODULES.d_sn = False
MODULES.d_bn = opt.batchnorm_2d
MODULES.d_linear = opt.linear
MODULES.d_act_fn = nn.ReLU(True)
MODULES.d_init = 'ortho'