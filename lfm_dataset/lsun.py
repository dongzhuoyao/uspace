import os
from pathlib import Path
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class LSUNBase(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.data_paths = Path(txt_file).expanduser().resolve()
        self.data_root = Path(data_root).expanduser().resolve()
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
        }

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = transforms.PILToTensor()(image)
        image = (image / 127.5 - 1.0).float()

        label_placeholder = np.array([1,2,3,4]).astype(np.uint8)
        return image, label_placeholder


class LSUNChurchesTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="~/data/lsun/church_outdoor_train.txt", data_root="~/data/lsun/churches_flat_train", **kwargs)


class LSUNChurchesValidation(LSUNBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="~/data/lsun/church_outdoor_val.txt", data_root="~/data/lsun/churches_flat_train",
                         flip_p=flip_p, **kwargs)


class LSUNBedroomsTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="~/data/lsun/bedrooms_train.txt", data_root="~/data/lsun/bedrooms", **kwargs)


class LSUNBedroomsValidation(LSUNBase):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(txt_file="~/data/lsun/bedrooms_val.txt", data_root="~/data/lsun/bedrooms",
                         flip_p=flip_p, **kwargs)


class LSUNCatsTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="~/data/lsun/cat_train.txt", data_root="~/data/lsun/cats", **kwargs)


class LSUNCatsValidation(LSUNBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="~/data/lsun/cat_val.txt", data_root="~/data/lsun/cats",
                         flip_p=flip_p, **kwargs)
