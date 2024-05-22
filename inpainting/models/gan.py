import torch.nn as nn
from external.model import _netG, _netlocalD


class Dis(nn.Module):
    """
    Discriminator for flow images of shape (batch, channel, 128, 128).
    Adds an initial conv2d layer.
    """

    def __init__(self, original_dis, opt):
        super(Dis, self).__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(opt.nc, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, opt.ndf, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(opt.ndf)
        )

        self.main = original_dis.main[1:-1]

        self.out = nn.Sequential(nn.Dropout(opt.Ddrop), nn.Sigmoid())

    def forward(self, input):
        x = self.initial_conv(input)
        x = self.main(x)
        x = self.out(x)
        return x.view(-1, 1)


class Gen(nn.Module):
    """
    Discriminator for flow images of shape (batch, channel, 128, 128).
    Adds a final convT2d layer. Removes tanh activation function.
    """

    def __init__(self, original_gen, opt):
        super(Gen, self).__init__()

        self.main = original_gen.main[:-2]
        self.last_uconv = nn.Sequential(
            nn.ConvTranspose2d(opt.ngf, 32, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, opt.nc, 4, 2, 1, bias=False),
        )

    def forward(self, input):
        x = self.main(input)
        x = self.last_uconv(x)
        return x


def build_gan(opt):
    netG_orig = _netG(opt)
    netG = Gen(netG_orig, opt)
    netD_orig = _netlocalD(opt)
    netD = Dis(netD_orig, opt)
    return netD, netG
