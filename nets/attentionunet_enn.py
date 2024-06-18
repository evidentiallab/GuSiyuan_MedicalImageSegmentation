import torch
import torch.nn as nn
from monai.networks.nets import AttentionUnet

from nets.DSFunction import Ds1


class AttentionUNetENN(nn.Module):
    def __init__(self, attentionunet_params):
        super(AttentionUNetENN, self).__init__()
        self.attentionunet = AttentionUnet(**attentionunet_params)
        self.evidential_layer = Ds1(2, 8, 2)

        for param in self.attentionunet.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.attentionunet(x)
        x = self.evidential_layer(x)
        return x

    def load_pretrained_attentionunet(self, weights_path):
        pretrained_weights = torch.load(weights_path)
        self.attentionunet.load_state_dict(pretrained_weights)
