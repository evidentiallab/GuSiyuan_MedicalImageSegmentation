import torch
import torch.nn as nn
from monai.networks.nets import SegResNet

from nets.DSFunction import Ds1


class SegResNetENN(nn.Module):
    def __init__(self, segresnet_params):
        super(SegResNetENN, self).__init__()
        self.segresnet = SegResNet(**segresnet_params)
        self.evidential_layer = Ds1(2, 15, 2)

        for param in self.segresnet.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.segresnet(x)
        x = self.evidential_layer(x)
        return x

    def load_pretrained_segresnet(self, weights_path):
        pretrained_weights = torch.load(weights_path)
        self.segresnet.load_state_dict(pretrained_weights)
