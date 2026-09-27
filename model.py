import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def create_model(num_classes=7):

    # Load ResNet18 pretrained on ImageNet
    model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    # Replace the original 1000-class ImageNet classifier
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(
            model.fc.in_features,
            num_classes
        )
    )

    return model