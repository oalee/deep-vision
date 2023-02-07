import torch as t
from torch import nn


class ResNetTuneModel(nn.Module):

    def __init__(self, resnet: nn.Module, num_classes: int, update_all_params: bool = False):
        super().__init__()

        self.resnet = resnet

        if not update_all_params:
            for param in self.resnet.parameters():
                param.requires_grad = False

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
