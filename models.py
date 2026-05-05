import torch
import torch.nn as nn
import torchvision.models as models


def get_model(name, num_classes, pretrained=True):
    name = name.lower()
    if name == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=weights)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
        return m
    if name == 'mobilenet_v2' or name == 'mobilenet':
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        m = models.mobilenet_v2(weights=weights)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)
        return m
    if name == 'efficientnet_b0' or name == 'efficientnet':
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b0(weights=weights)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)
        return m
    if name == 'simplecnn':
        class SimpleCNN(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3,32,3,1,1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32,64,3,1,1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.AdaptiveAvgPool2d(1)
                )
                self.classifier = nn.Linear(64, num_classes)
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        return SimpleCNN(num_classes)
    raise ValueError('Unknown model: '+name)
