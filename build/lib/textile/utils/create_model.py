import torch
from torch import nn
from torchvision.models import convnext_base

from textile.architectures.layers.attention.attention import LinearAttention


def CreateModel(pre_trained_network=None):
    """
    Creates model with pre-trained weights.
    :param pre_trained_network: Path to pre-trained weights
    :return: Textile model
    """

    model = convnext_base(weights=None)

    model.classifier = nn.Sequential(
        nn.Flatten(1), nn.Linear(1024, 1)

    )

    # Note that we are introducing self-attention layers to the Convnext architecture
    layers = []

    layers.append(model.features[0])
    layers.append(model.features[1])
    layers.append(model.features[2])
    layers.append(model.features[3])
    layers.append(model.features[4])
    layers.append(LinearAttention(512, 16, 128))
    layers.append(model.features[5])
    layers.append(model.features[6])
    layers.append(LinearAttention(1024, 16, 256))
    layers.append(model.features[7])
    model.features = nn.Sequential(*layers)

    if pre_trained_network is not None:
        # print("Reading model from disc")
        model_weights = torch.load(pre_trained_network)
        model.load_state_dict(model_weights, strict=False)
    return model
