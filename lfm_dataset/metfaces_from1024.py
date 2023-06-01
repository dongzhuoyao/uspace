# modified from https://github.com/CompVis/taming-transformers/blob/24268930bf1dce879235a7fddd0b2355b84d7ea6/taming/data/base.py#L23
# https://github.com/CompVis/taming-transformers#ffhq

import json
import math
import os
from pathlib import Path
import PIL
from einops import rearrange
from loguru import logger
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision import transforms
from numpy.random import default_rng
import h5py




class MetFaces_From1024(Dataset):
    def __init__(
        self,
        root,
        size=None,
        debug=False,
    ):
        #https://github.com/NVlabs/metfaces-dataset

        self.dataset_name = "metfaces"
        self.debug = debug
        self.size  = size

        root = Path(root).expanduser().resolve()
        
        logger.warning(f"reading from dir {root}")
        pathlist = list()

        
        for name in os.listdir(root):
            if name.endswith(".png"):
                pathlist.append(os.path.join(root, name))

        self.pathlist = pathlist
        self._length = len(self.pathlist)
        print("self._length", self._length)
        assert self._length == 1336, f"expected 1336 images, got {self._length}"


   

    def _read_img(self, index):
        image_path = self.pathlist[index]
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        return image

    def __len__(self):
        return 1000 if self.debug else self._length


    def __getitem__(self, index):
      
        image = self._read_img(index)
        if False:
            image = torch.from_numpy(
                np.array(image.resize((self.size, self.size), Image.BILINEAR))
            )
        else:
            image = torch.from_numpy(np.array(image))

        image = rearrange(image, "w h c -> c w h")
        image = (image / 255.0)* 2 - 1  # [0,1]->[-1,1]

        
        return image, image 


def resize_to_256(root, size=256):
    root_new = Path(root + f"_size{size}")
    if not root_new.exists():
        os.mkdir(root_new)
        print(f"created {root_new}")

    for name in tqdm(os.listdir(root)):
        if name.endswith(".png"):
            image_path = os.path.join(root, name)
            image = Image.open(image_path)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image = image.resize((size, size), Image.BILINEAR)
            image.save(os.path.join(root_new, name))


if __name__ == "__main__":
    resize_to_256(root="/home/thu/data_hhd/metfaces/images")

