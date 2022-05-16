import torch.nn as nn

class BasicSRModel(nn.Module):
    def __init__(self, upscale_factor = 2, layers = 10, residual = False) :
        super(BasicSRModel, self).__init__()
        self.up_sample = nn.Upsample(scale_factor=upscale_factor, mode='bilinear')
        self.conv_first = nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))
        self.conv_last = nn.Conv2d(64, 3, (3, 3), (1, 1), (1, 1))
        modules = []
        for i in range(layers):
            modules.append(nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)))
            modules.append(nn.LeakyReLU())
        self.conv_middle = nn.Sequential(*modules)
        self.residual = residual

    def forward(self, x):
        x_up = self.up_sample(x)
        out1 = self.conv_first(x_up)
        out2 = self.conv_middle(out1)
        out = self.conv_last(out2)
        if self.residual:
            out += x_up
        return out