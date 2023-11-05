import torch
import torch.nn as nn
from monai.networks.nets import VNet

from DSFunction import Ds1


class VNetENN(nn.Module):
    def __init__(self, vnet_params):
        super(VNetENN, self).__init__()
        self.vnet = VNet(**vnet_params)
        self.evidential_layer = Ds1(2, 10, 2)

        for param in self.vnet.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.vnet(x)
        x = self.evidential_layer(x)
        return x

    def load_pretrained_unet(self, weights_path):
        pretrained_weights = torch.load(weights_path)
        self.vnet.load_state_dict(pretrained_weights)
