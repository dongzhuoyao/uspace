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



class FFHQ_v2(Dataset):
    def __init__(
        self,
        root,
        size=None,
        size_file = 1024,
        split="train",
        img_save_path=None,
        num_samples=1.0,
        random_crop=False,
        debug=False,
        seed=0,
    ):

        self.dataset_name = "ffhq"
        self.img_save_path = img_save_path
        self.debug = debug
        self.split_name = split
        self.size  = size
        self.random_crop = random_crop
        

        if size_file == 64:
            root = root.replace("128", "64")
            print("root = root.replace('128','64')")
        else:
            assert "128" in root or "1024" in root, "root {} must contain 128/1024".format(root)
        logger.warning(f"reading from dir {root}")

        if split == "train":
            txt_name = "lfm_dataset/data_files/ffhqtrain.txt"
        else:
            txt_name = "lfm_dataset/data_files/ffhqvalidation.txt"

        pathlist = list()
        with open(txt_name, "r") as f:
            relpaths = f.read().splitlines()
        for name in relpaths:
            _file_name = (
                str(int(name.replace(".png", "")) //
                    1000).zfill(2) + "000/" + name
            )
            pathlist.append(
                str(Path(os.path.join(root, _file_name)).expanduser().resolve()))
        #################
         
        self.pathlist = pathlist

        if num_samples is not None:

            idx = np.array([i for i in range(len(self.pathlist))])
            default_rng(seed).shuffle(idx)

            self.pathlist = [self.pathlist[_id] for _id in idx]

            if isinstance(num_samples, int):
                _partial_rate = num_samples / len(self.pathlist)

            elif isinstance(num_samples, float):
                _partial_rate = num_samples
                num_samples = int(_partial_rate * len(self.pathlist))

            else:
                raise ValueError(
                    f'num_samples must be int or float, got {type(num_samples)}')
            self.pathlist = self.pathlist[:num_samples]
                
            logger.warning(f'Using only {num_samples} images')
            _partial_rate_inverse = math.ceil(1.0/_partial_rate)

            self.pathlist = self.pathlist*_partial_rate_inverse
            


        self._length = len(self.pathlist)


    def id2name(self, index):
        file_name = os.path.basename(self.pathlist[index])
        return file_name

    def _read_img_segmask(self, index):
        image_path = self.pathlist[index]
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        return image, None

    def __len__(self):
        return 1000 if self.debug else self._length

    def get_imgid_from_imagepath(self, image_path):
        return os.path.basename(image_path).split(".")[0]

    def __getitem__(self, index):
        result = dict()
        image, _ = self._read_img_segmask(index)
        image = torch.from_numpy(
            np.array(image.resize((self.size, self.size), Image.BILINEAR))
        )
        image = rearrange(image, "w h c -> c w h")

        image = (image / 255.0)* 2 - 1  # [0,1]->[-1,1]

        
        return image, image 


if __name__ == "__main__":
    pass
