import os
from pathlib import Path
import re
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from PIL import Image
import os, torchvision

try:
    from .celeba import _find_images_and_annotation
except:
    from celeba import _find_images_and_annotation


def vis_parsing_maps(
    im,
    parsing_anno,
    stride,
    save_im=False,
    save_path="vis_results/parsing_map_on_im.jpg",
):
    # Colors for all 20 parts
    part_colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 0, 85],
        [255, 0, 170],
        [0, 255, 0],
        [85, 255, 0],
        [170, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [0, 85, 255],
        [0, 170, 255],
        [255, 255, 0],
        [255, 255, 85],
        [255, 255, 170],
        [255, 0, 255],
        [255, 85, 255],
        [255, 170, 255],
        [0, 255, 255],
        [85, 255, 255],
        [170, 255, 255],
    ]

    im = np.array(im)
    parsing_anno = np.array(parsing_anno)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    # vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = (
        np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    )

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(
        cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0
    )

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] + ".png", vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        cv2.imwrite(
            save_path.replace(".jpg", "_origin.jpg"),
            im,
            [int(cv2.IMWRITE_JPEG_QUALITY), 100],
        )


class CelebAMaskHQ256:
    def __init__(self, data_path, root_celebA, n_bits=8, image_size=256, mode=True):
        data_path = Path(data_path).expanduser().resolve()
        root_celebA = Path(root_celebA).expanduser().resolve()

        self.img_path = os.path.join(data_path, "CelebA-HQ-img")
        self.label_path = os.path.join(data_path, "mask")
        self.map2CelebAAttr = os.path.join(data_path, "mask")
        self.celebmask_celeba_mapping_txt = os.path.join(
            data_path, "CelebA-HQ-to-CelebA-mapping.txt"
        )

        self.root_celebA = root_celebA  #'celeba_torchvision/celeba/'

        self.mask_name_list = [
            "class0_placeholder",
            "skin",
            "l_brow",
            "r_brow",
            "l_eye",
            "r_eye",
            "eye_g",
            "l_ear",
            "r_ear",
            "ear_r",
            "nose",
            "mouth",
            "u_lip",
            "l_lip",
            "neck",
            "neck_l",
            "cloth",
            "hair",
            "hat",
        ]

        self.n_bits = n_bits

        def preprocess(x):
            # Follows:
            # https://github.com/tensorflow/tensor2tensor/blob/e48cf23c505565fd63378286d9722a1632f4bef7/tensor2tensor/models/research/glow.py#L78

            x = x * 255  # undo ToTensor scaling to [0,1]

            n_bins = 2**n_bits
            if n_bits < 8:
                x = torch.floor(x / 2 ** (8 - n_bits))
            x = x / n_bins

            return x

        def postprocess(x):
            x = torch.clamp(x, 0, 1)
            x = x * 2**8
            return torch.clamp(x, 0, 255).byte()

        assert image_size in [256, 64]
        transform_img = transforms.Compose(
            [
                # transforms.CenterCrop(512),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                preprocess,
            ]
        )

        transform_label = transforms.Compose(
            [
                # transforms.CenterCrop(512),
                transforms.Resize(
                    image_size,
                    interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                ),
                transforms.PILToTensor(),
            ]
        )

        self.transform_img = transform_img
        self.transform_label = transform_label
        self.train_dataset = []
        self.test_dataset = []
        self.mode = mode
        self.preprocess()

        if mode == True:
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

        print("begin _find_images_and_annotation")
        self.celeba_dicts, self.celeba_attrs = _find_images_and_annotation(
            self.root_celebA
        )
        print("end _find_images_and_annotation")
        self.attrs = self.celeba_attrs
        self.celebname2attr = dict()
        self.celebmaskname2attr = dict()
        for l in self.celeba_dicts:
            path, attr = l["path"], l["attr"]
            self.celebname2attr[path.split("/")[-1].strip()] = attr  # 000096.jpg

        with open(self.celebmask_celeba_mapping_txt, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.replace("\n", "").split()
                celebmask_id = line[0].strip()
                celeba_id = line[1].strip()
                celeba_filename = line[2].strip()
                self.celebmaskname2attr[celebmask_id + ".jpg"] = self.celebname2attr[
                    celeba_filename
                ]

    def preprocess(self):
        for i in range(
            len(
                [
                    name
                    for name in os.listdir(self.img_path)
                    if os.path.isfile(os.path.join(self.img_path, name))
                ]
            )
        ):
            img_path = os.path.join(self.img_path, str(i) + ".jpg")
            label_path = os.path.join(self.label_path, str(i) + ".png")
            # print(img_path, label_path)
            if self.mode == True:
                self.train_dataset.append([img_path, label_path])
            else:
                self.test_dataset.append([img_path, label_path])

        print("Finished preprocessing the CelebA dataset...")

    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode == True else self.test_dataset

        img_path, label_path = dataset[index]

        image, _segmask = Image.open(img_path), Image.open(label_path).convert(
            "P"
        )  # https://github.com/zllrunning/face-parsing.PyTorch/blob/d2e684cf15/face_dataset.py#L48

        image, _segmask = self.transform_img(image), self.transform_label(_segmask)

        image = image * 2 - 1  # [-1,1]

        if True:
            img_name = img_path.split("/")[-1].strip()
            attr = self.celebmaskname2attr[img_name]
            return image, _segmask, np.asarray(attr)
        else:
            return image, _segmask

    def __len__(self):
        """Return the number of images."""
        return self.num_images


if __name__ == "__main__":
    import cv2

    celeba = torch.utils.data.DataLoader(
        CelebAMaskHQ256(
            mode=True,
            data_path="~/data/celebamask_hd/CelebAMask-HQ",
            root_celebA="~/data/celeba_torchvision/celeba",
        ),
        batch_size=4,
    )

    d = next((iter(celeba)))
    for _ii, d in enumerate(celeba):
        img = d["x"][0]
        print(img.size())
        img = img * 256
        img = img.permute(1, 2, 0).contiguous().numpy()
        # img = np.array(transforms.ToPILImage()(img))
        img = img[:, :, ::-1].copy()  # need extra step
        # print("img", np.min(img), np.max(img))

        label = d["seg"][0]
        label = label.permute(1, 2, 0).contiguous().numpy()
        # label = transforms.ToPILImage()(label)
        # print("label", np.min(label), np.max(label))

        vis_parsing_maps(
            im=img,
            parsing_anno=label,
            stride=1,
            save_im=True,
            save_path="daemo{}.jpg".format(_ii),
        )
        print("aa")
        # cv2.imshow("img", img)
        # cv2.imshow("label", label)
        # cv2.waitKey()
