import torch

from ERA import get_rational_parameters, ERA
from torch import nn


class VGG(nn.Module):
    def __init__(self, era_args):
        super().__init__()
        numerator, denominator = get_rational_parameters(era_args)

        self.conv1 = nn.Conv2d(3, 64, 3)
        self.era1 = ERA(numerator, denominator, era_args, [32, 32, 64], [1, 2], [1, 2])
