import sys,os
path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)

from lfm_dataset.celebamask30k_1024 import CelebAMaskHQ256
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader
from libs.autoencoder import get_model
import argparse
from tqdm import tqdm
torch.manual_seed(0)
np.random.seed(0)


def main(resolution=256):
    parser = argparse.ArgumentParser()
    

    dataset = CelebAMaskHQ256(data_path="~/data/celebamask_hd/CelebAMask-HQ", root_celebA="~/data/celeba_torchvision/celeba",image_size=256, n_bits=8)
    
    train_dataset_loader = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False,
                                      num_workers=8, pin_memory=True, persistent_workers=True)

    model = get_model('assets/stable-diffusion/autoencoder_kl.pth')
    model = nn.DataParallel(model)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # features = []
    # labels = []

    idx = 0
    for batch in tqdm(train_dataset_loader):
        img, label = batch
        #img = torch.cat([img, img.flip(dims=[-1])], dim=0)
        #label = torch.cat([label, label], dim=0)

        img = img.to(device)
        moments = model(img, fn='encode_moments')
        moments = moments.detach().cpu().numpy()

        
        label = label.detach().cpu().numpy()

        for moment, lb in zip(moments, label):
            np.save(f'assets/datasets/celebamask{resolution}_features/{idx}.npy', (moment, lb))
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
