import torch
import os
from model import create_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():

    model = create_model()

    model_path = os.path.join(
        os.path.dirname(__file__),
        "best_resnet101.pth"
    )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)

    model.to(DEVICE)

    model.eval()

    return model