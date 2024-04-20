import os

import numpy as np
import torch
from torch import nn

from textile.utils.create_model import CreateModel


class Textile(nn.Module):
    def __init__(self, model_path: str = "models/textile.pth", lambda_value: float = 0.25):
        """
        Implementation of TexTile: A Differentiable Metric for Texture Tileability
        :param model_path: Path to pretrained model
        :param lambda_value: Lambda value to transform the unbounded model prediction to the (0, 1) range. Higher lambdas provide more sensitive predictions. Check our supplementary material for more details.
        """
        super(Textile, self).__init__()

        assert torch.cuda.is_available()
        assert model_path.endswith('.pth')
        assert lambda_value >= 0 and lambda_value <= 1

        is_model_on_disc = os.path.exists(model_path)
        if not is_model_on_disc:
            try:
                import urllib.request
                urllib.request.urlretrieve("http://www.example.com/songs/mp3.mp3", "models/textile.pth")
            except Exception as e:
                print('Could not retrieve pretrained model')
                raise e

        self.model = CreateModel(model_path)
        self.lambda_value = torch.tensor(lambda_value)

    def forward(self, image: torch.Tensor, return_logits: bool = False):
        """
        Forward function
        :param image: Tiled image
        :param return_numpy: Set to true if you want the raw, unbounded, model logits (For optimization purposes).
        :return: Estimated textile prediction
        """
        assert image.dim() == 4
        assert image.size(1) == 3

        result = self.model(image.float().cuda())
        if return_logits is False:
            result = 1 / (1 + torch.exp((-self.lambda_value * result)))

        return result
