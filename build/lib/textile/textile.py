from __future__ import absolute_import


import os
from torchvision import transforms

import numpy as np
import torch
from torch import nn

import textile

from textile.utils.misc import MyProgressBar
from textile.utils.create_model import CreateModel


class Textile(nn.Module):
    def __init__(self, model_path: str = "textile/models/textile.pth", lambda_value: float = 0.25, resolution = (512, 512), number_tiles = 2):
        """
        Implementation of TexTile: A Differentiable Metric for Texture Tileability
        :param model_path: Path to pretrained model
        :param lambda_value: Lambda value to transform the unbounded model prediction to the (0, 1) range. Higher lambdas provide more sensitive predictions. Check our supplementary material for more details.
        :param resolution: Resolution of the image provided to the model after tiling.
        :param number_tiles: Number of tiles to tile the image
        """
        super(Textile, self).__init__()

        assert torch.cuda.is_available()
        assert model_path.endswith('.pth')
        assert lambda_value >= 0 and lambda_value <= 1


        is_model_on_disc = os.path.exists(model_path)
        if not is_model_on_disc:
            try:
                import urllib.request
                from pathlib import Path
                print('Model not found, downloading pretrained model:')
                Path("textile/models/").mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve("https://carlosrodriguezpardo.es/projects/TexTile/models/textile_v3.pth", model_path, MyProgressBar())
            except Exception as e:
                print('Could not retrieve pretrained model')
                raise e

        self.model = CreateModel(model_path).cuda().eval()
        self.lambda_value = torch.tensor(lambda_value)
        self.t_resized = transforms.Resize(resolution, antialias=True)
        self.transform = nn.Sequential(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )
        self.number_tiles = number_tiles

    def forward(self, image: torch.Tensor, return_logits: bool = False, normalize = True, rescale = True, tile = True):
        """
        Forward function
        :param image: Tiled image
        :param return_numpy: Set to true if you want the raw, unbounded, model logits (For optimization purposes).
        :param normalize: Set to true if you want to normalize the image (Only set to false if you already normalized it)
        :param rescale: Set to true if you want to rescale the tiled image to the appropriate resolution
        :param tile: Set to false if you do not want to tile the image
        :return: Estimated textile prediction
        """
        assert image.dim() == 4
        assert image.size(1) == 3

        if tile:
            image = torch.tile(image, (1, 1, self.number_tiles, self.number_tiles))

        if rescale:
            image = self.t_resized.forward(image)

        if normalize:
            image = self.transform(image)

        result = self.model(image.float().cuda())
        if return_logits is False:
            result = 1 / (1 + torch.exp((-self.lambda_value * result)))

        return result
