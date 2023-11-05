import torch
import torch.nn as nn
from monai.networks.nets import UNet

from nets.DSFunction import Ds1


class UNetENN(nn.Module):
    def __init__(self, unet_params):
        super(UNetENN, self).__init__()
        self.unet = UNet(**unet_params)
        self.evidential_layer = Ds1(2, 10, 2)

        for param in self.unet.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.unet(x)
        x = self.evidential_layer(x)
        return x

    def load_pretrained_unet(self, weights_path):
        pretrained_weights = torch.load(weights_path)
        self.unet.load_state_dict(pretrained_weights)
