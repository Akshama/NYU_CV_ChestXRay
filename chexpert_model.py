import torch
from torch import nn
from chexpert_data import NUM_CLASSES


class Densenet121(nn.Module):
    def __init__(self, flag):
        super().__init__()
        self.base = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=flag)
        self.age = nn.Linear(1000, 1)
        self.sex = nn.Linear(1000, 1)
        self.angle = nn.Linear(1000, 1)
        self.pathologies = nn.Linear(1000, NUM_CLASSES)

    def forward(self, xs):
        batch_size = xs.size(0)
        emb = self.base(xs)
        return (self.sex(emb), self.age(emb), self.angle(emb), self.pathologies(emb))


class ResNet152(nn.Module):
    def __init__(self, flag):
        super().__init__()
        self.base = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=flag)
        self.age = nn.Linear(1000, 1)
        self.sex = nn.Linear(1000, 1)
        self.angle = nn.Linear(1000, 1)
        self.pathologies = nn.Linear(1000, NUM_CLASSES)

    def forward(self, xs):
        batch_size = xs.size(0)
        emb = self.base(xs)
        return (self.sex(emb), self.age(emb), self.angle(emb), self.pathologies(emb))
