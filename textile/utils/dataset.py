import cv2
import kornia as K
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms


class TilingDataset(Dataset):
    def __init__(
            self,
            images_tileable,
            resolution=(512, 512),
            number_tiles=2,
    ):
        self.images_tileable = images_tileable
        self.length = len(images_tileable)
        self.resolution = resolution
        self.number_tiles = number_tiles
        self.t_resized = transforms.Resize(resolution, antialias=True)
        self.transform = nn.Sequential(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < len(self.images_tileable):
            image = self.images_tileable[idx]
            image = cv2.imread(image, -1)[:, :, :3][:, :, ::-1] / 255
            image = K.image_to_tensor(image)
            image = torch.tile(image, (1, self.number_tiles, self.number_tiles))
            image = self.t_resized.forward(image.unsqueeze(0)).squeeze()
            image = self.transform(image)
        return image
