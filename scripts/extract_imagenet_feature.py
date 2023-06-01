import sys,os
path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from libs.autoencoder import get_model
from datasets import ImageNet
from tqdm import tqdm
import argparse



torch.manual_seed(0)
np.random.seed(0)


def main(resolution=256):
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()

    dataset = ImageNet(
        path=args.path, resolution=resolution, random_flip=False)
    train_dataset = dataset.get_split(split='train', labeled=True)
    train_dataset_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, drop_last=False,
                                      num_workers=8, pin_memory=True, persistent_workers=True)

    model = get_model('assets/stable-diffusion/autoencoder_kl.pth')
    model = nn.DataParallel(model)
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # features = []
    # labels = []

    idx = 0
    for batch in tqdm(train_dataset_loader):
        img, label = batch
        img = torch.cat([img, img.flip(dims=[-1])], dim=0)
        img = img.to(device)
        moments = model(img, fn='encode_moments')
        moments = moments.detach().cpu().numpy()

        label = torch.cat([label, label], dim=0)
        label = label.detach().cpu().numpy()

        for moment, lb in zip(moments, label):
            np.save(
                f'assets/datasets/imagenet{resolution}_features/{idx}.npy', (moment, lb))
            idx += 1

    print(f'save {idx} files')

    # features = np.concatenate(features, axis=0)
    # labels = np.concatenate(labels, axis=0)
    # print(f'features.shape={features.shape}')
    # print(f'labels.shape={labels.shape}')
    # np.save(f'imagenet{resolution}_features.npy', features)
    # np.save(f'imagenet{resolution}_labels.npy', labels)


if __name__ == "__main__":
    main()
