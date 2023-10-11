import torch
from torch import nn
from monai.networks.nets import UNet
from monai.networks.layers import Norm


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = UNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    net = Net()
    input = torch.ones((1, 2, 256, 512, 512))
    output = net(input)
    print(output.shape)
