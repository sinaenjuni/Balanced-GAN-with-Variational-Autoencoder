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

MODULES.g_conv2d = opt.snconv2d
MODULES.g_sn = True
MODULES.g_bn = opt.ConditionalBatchNorm2d
# MODULE.g_act_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
MODULES.g_linear = opt.snlinear
MODULES.g_act_fn = nn.ReLU(True)
MODULES.g_init = 'ortho'

MODULES.d_conv2d = opt.snconv2d
MODULES.d_linear = opt.snlinear
MODULES.d_embedding = opt.sn_embedding
MODULES.d_sn = True
MODULES.d_bn = opt.batchnorm_2d
MODULES.d_act_fn = nn.ReLU(True)
MODULES.d_init = 'ortho'