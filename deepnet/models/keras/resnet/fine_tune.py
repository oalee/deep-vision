import torch as t

from torch import nn


class ResNetTuneModel(nn.Module):

    def __init__(self, num_classes, resnet: nn.Module):
        super().__init__()

        self.resnet = resnet

        requires_grad = False
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Replace the last fully connected layer
        # Parameters of newly constructed modules have requires_grad=True by default
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
