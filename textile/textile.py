from __future__ import absolute_import


import os
from torchvision import transforms

import numpy as np
import torch
from torch import nn

import textile

from textile.utils.create_model import CreateModel

HF_REPO_ID = "crp94/textile"
HF_FILENAME = "model.safetensors"


class Textile(nn.Module):
    def __init__(self, model_path: str = None, lambda_value: float = 0.25, resolution = (512, 512), number_tiles = 2):
        """
        Implementation of TexTile: A Differentiable Metric for Texture Tileability
        :param model_path: Path to pretrained model (.safetensors or .pth). If None, it is downloaded from the Hugging Face Hub.
        :param lambda_value: Lambda value to transform the unbounded model prediction to the (0, 1) range. Higher lambdas provide more sensitive predictions. Check our supplementary material for more details.
        :param resolution: Resolution of the image provided to the model after tiling.
        :param number_tiles: Number of tiles to tile the image
        """
        super(Textile, self).__init__()

        assert torch.cuda.is_available()
        assert lambda_value >= 0 and lambda_value <= 1

        if model_path is None:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
        elif not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Pass model_path=None to download it from https://huggingface.co/{HF_REPO_ID}")

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
