import torch.nn as nn
from torchvision.models import resnet101


def create_model():

    model = resnet101(weights=None)

    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        5
    )

    return model