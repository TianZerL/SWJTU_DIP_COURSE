import torch as tc
from torch import nn


class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()
        self.sequential = None
        self.pad = None

    def forward(self, x):
        if not (self.pad is None):
            x = self.pad(x)
        return self.sequential.forward(x)


class Upconv7RGB(BaseModule):
    def __init__(self):
        super(Upconv7RGB, self).__init__()
        self.actFunc = nn.LeakyReLU(0.1, inplace=False)
        self.offset = 7
        self.pad = nn.ZeroPad2d(self.offset)
        m = [
            nn.Conv2d(3, 16, 3, 1, 0),
            self.actFunc,
            nn.Conv2d(16, 32, 3, 1, 0),
            self.actFunc,
            nn.Conv2d(32, 64, 3, 1, 0),
            self.actFunc,
            nn.Conv2d(64, 128, 3, 1, 0),
            self.actFunc,
            nn.Conv2d(128, 128, 3, 1, 0),
            self.actFunc,
            nn.Conv2d(128, 256, 3, 1, 0),
            self.actFunc,
            nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=3, bias=False),
        ]
        self.sequential = nn.Sequential(*m)


class FSRCNN(BaseModule):
    def __init__(self):
        super(FSRCNN, self).__init__()
        self.actFunc = nn.PReLU()
        m = [
            nn.Conv2d(1, 56, 5, 1, 2),
            self.actFunc,
            nn.Conv2d(56, 12, 1, 1, 0),
            self.actFunc,
            nn.Conv2d(12, 12, 3, 1, 1),
            self.actFunc,
            nn.Conv2d(12, 12, 3, 1, 1),
            self.actFunc,
            nn.Conv2d(12, 12, 3, 1, 1),
            self.actFunc,
            nn.Conv2d(12, 12, 3, 1, 1),
            self.actFunc,
            nn.Conv2d(12, 56, 1, 1, 0),
            self.actFunc,
            nn.ConvTranspose2d(
                56, 1, kernel_size=9, stride=2, padding=4, output_padding=1
            ),
        ]
        self.sequential = nn.Sequential(*m)


class ACNet(BaseModule):
    def __init__(self):
        super(ACNet, self).__init__()
        self.actFunc = nn.ReLU()
        m = [
            nn.Conv2d(1, 8, 3, 1, 1),
            self.actFunc,
            nn.Conv2d(8, 8, 3, 1, 1),
            self.actFunc,
            nn.Conv2d(8, 8, 3, 1, 1),
            self.actFunc,
            nn.Conv2d(8, 8, 3, 1, 1),
            self.actFunc,
            nn.Conv2d(8, 8, 3, 1, 1),
            self.actFunc,
            nn.Conv2d(8, 8, 3, 1, 1),
            self.actFunc,
            nn.Conv2d(8, 8, 3, 1, 1),
            self.actFunc,
            nn.Conv2d(8, 8, 3, 1, 1),
            self.actFunc,
            nn.Conv2d(8, 8, 3, 1, 1),
            self.actFunc,
            nn.ConvTranspose2d(8, 1, kernel_size=2, stride=2, padding=0, bias=False),
        ]
        self.sequential = nn.Sequential(*m)


modules = {
    "Waifu2x": Upconv7RGB(),
    "FSRCNN": FSRCNN(),
    "ACNet": ACNet(),
}
moduleNames = {
    "Waifu2x": "Upconv7RGB.ptm",
    "FSRCNN": "FSRCNN.ptm",
    "ACNet": "ACNet.ptm",
}
moduleColorFormat = {
    "Waifu2x": "RGB",
    "FSRCNN": "L",
    "ACNet": "L",
}
