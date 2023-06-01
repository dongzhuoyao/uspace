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


class FFHQ_From1024(Dataset):
    def __init__(
        self,
        root,
        att_root="~/data/ffhq-features-dataset/json",
        size=None,
        split="train",
        random_crop=False,
        debug=False,
    ):
        self.dataset_name = "ffhq"
        self.debug = debug
        self.split_name = split
        self.size = size
        self.random_crop = random_crop
        self.att_root = str(Path(att_root).expanduser().resolve())

        logger.warning(f"reading from dir {root}")

        if split == "train":
            txt_name = "lfm_dataset/data_files/ffhqtrain.txt"
        else:
            txt_name = "lfm_dataset/data_files/ffhqvalidation.txt"

        pathlist = list()
        with open(txt_name, "r") as f:
            relpaths = f.read().splitlines()
        for name in relpaths:
            pathlist.append(str(Path(os.path.join(root, name)).expanduser().resolve()))

        self.pathlist = pathlist
        self.pathlist = self.filter_path(self.pathlist)
        self._length = len(self.pathlist)

    def id2name(self, index):
        file_name = os.path.basename(self.pathlist[index])
        return file_name

    def _read_img_segmask(self, index):
        image_path = self.pathlist[index]
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        return image, None, self.get_imgid_from_imagepath(image_path)

    def __len__(self):
        return 1000 if self.debug else self._length

    def get_imgid_from_imagepath(self, image_path):
        return os.path.basename(image_path).split(".")[0]

    def __getitem__(self, index):
        image, _, image_id = self._read_img_segmask(index)
        _attr = self.load_attr_by_image_id(image_idstr=str(image_id))
        if _attr is not None:
            pass
            # self.imgid2attr[image_id] = _attr
        image = torch.from_numpy(
            np.array(image.resize((self.size, self.size), Image.BILINEAR))
        )
        image = rearrange(image, "w h c -> c w h")

        image = (image / 255.0) * 2 - 1  # [0,1]->[-1,1]

        return image, torch.from_numpy(np.array(_attr))

    def filter_path(self, _paths):
        _new_paths = []
        for _path in _paths:
            try:
                image_idstr = self.get_imgid_from_imagepath(_path)
                json_file = json.load(
                    open(os.path.join(self.att_root, image_idstr + ".json"), "r")
                )
                faceAttributes = json_file[0]["faceAttributes"]
                _new_paths.append(_path)
            except:
                pass
                # print("skip", _path)
        print("filter_path done, old len: ", len(_paths), "new len: ", len(_new_paths))
        return _new_paths

    def load_attr_by_image_id(self, image_idstr):
        try:
            json_file = json.load(
                open(os.path.join(self.att_root, image_idstr + ".json"), "r")
            )
            faceAttributes = json_file[0]["faceAttributes"]
        except:
            # print(f'shit happens, {image_idstr}')
            raise
            return None

        gender = 0 if faceAttributes["gender"] == "female" else 1
        smile = 1 if faceAttributes["smile"] > 0.5 else 0
        no_glasses = 1 if faceAttributes["glasses"] == "NoGlasses" else 0  # remove
        ########
        emotion = faceAttributes["emotion"]
        anger = 1 if emotion["anger"] > 0.5 else 0
        contempt = 1 if emotion["contempt"] > 0.5 else 0
        disgust = 1 if emotion["disgust"] > 0.5 else 0
        fear = 1 if emotion["fear"] > 0.5 else 0
        happiness = 1 if emotion["happiness"] > 0.5 else 0
        neutral = 1 if emotion["neutral"] > 0.5 else 0
        sadness = 1 if emotion["sadness"] > 0.5 else 0
        surprise = 1 if emotion["surprise"] > 0.5 else 0
        return [
            gender,
            smile,
            no_glasses,
            anger,
            contempt,
            disgust,
            fear,
            happiness,
            neutral,
            sadness,
            surprise,
        ]


if __name__ == "__main__":
    dataset = FFHQ_From1024(root="~/data/ffhq/thumbnails64x64_onelevel", size=256)
    for i in range(len(dataset)):
        _img, _attr = dataset.__getitem__(i)
        print(_img.shape, _attr.shape)
        break
