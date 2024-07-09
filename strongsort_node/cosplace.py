import numpy as np

import os
from os.path import join, exists, isfile, realpath, dirname
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import sys
import pickle
import sklearn
from sklearn.neighbors import NearestNeighbors

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class CosPlace(object):
    """CosPlace matcher
    """

    def __init__(self):
        """Initialization

        Args:
            params (dict): parameters
        """
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = torch.hub.load("gmberton/cosplace", "get_trained_model", backbone="ResNet18", fc_output_dim=64)
        self.model.to(self.device)

        self.model.eval()
        self.transform = transforms.Compose([
            transforms.CenterCrop(376),
            transforms.Resize(224, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN,
                                    IMAGENET_DEFAULT_STD),
        ])
            

    def compute_embedding(self, keyframe):
        """Load image to device and extract the global image descriptor

        Args:
            keyframe (image): image to match

        Returns:
            np.array: global image descriptor
        """
        with torch.no_grad():
            # print("Device:", self.device)
            # print("Transforms:", self.transforms)
            # print("Image", image)

            image = Image.fromarray(keyframe)
            input = self.transform(image)
            input = torch.unsqueeze(input, 0)
            input = input.to(self.device)

            image_encoding = self.model.forward(input)

            output = image_encoding[0].detach().cpu().numpy()
            del input, image_encoding, image

        return output