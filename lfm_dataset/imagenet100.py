import os, yaml, pickle, shutil, tarfile, glob
from pathlib import Path
import PIL
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import blobfile as bf
from torchvision import transforms


class ImageNet100(Dataset):
    def __init__(
        self,
        root="~/data/imagenet/train",
        size=256,
        file_path="./imagenet100.txt",
    ):
        self.size = size

        file_path = os.path.join(os.path.dirname(__file__), file_path)
        with open(file_path, "rb") as f:
            self.folder_list = f.readlines()

        root = Path(root).expanduser().resolve()

        self.folder_list = [x.decode("utf-8").strip() for x in self.folder_list]
        self.data_path_list = []
        self.label_list = []
        for cls_id, _folder in enumerate(self.folder_list):
            _folder = os.path.join(root, _folder)
            assert os.path.exists(_folder), f"{_folder} does not exist"
            for _image_name in os.listdir(_folder):
                image_path = os.path.join(_folder, _image_name)
                assert os.path.exists(image_path), f"{image_path} does not exist"
                self.data_path_list.append(image_path)
                self.label_list.append(cls_id)

        self.label_list = np.array(self.label_list)

        print("ImageNet100k data_path_list", len(self.data_path_list))

    def __len__(self):
        if False:
            return 1000
        return len(self.data_path_list)

    def __getitem__(self, i):
        img_path, _label = self.data_path_list[i], self.label_list[i]

        with bf.BlobFile(img_path, "rb") as f:
            _image = Image.open(f).convert("RGB")
            _image.load()

        # default to score-sde preprocessing
        _image = np.array(_image).astype(np.uint8)
        crop = min(_image.shape[0], _image.shape[1])
        (
            h,
            w,
        ) = (
            _image.shape[0],
            _image.shape[1],
        )
        _image = _image[
            (h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2
        ]

        _image = Image.fromarray(_image)
        if self.size is not None:
            _image = _image.resize(
                (self.size, self.size), resample=Image.Resampling.BICUBIC
            )
        _image = transforms.ToTensor()(_image)
        _image = 2 * _image - 1
        # print(_label)
        return _image, np.array([_label], dtype=np.int64)


if __name__ == "__main__":
    dataset = ImageNet100()
    print(len(dataset))
    image, label = dataset.__getitem__(3)
    print(image.shape, label.shape)
