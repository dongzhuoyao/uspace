import sys, os

path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)

import torch
import os
import numpy as np
import libs.autoencoder
import libs.clip
from datasets import MMCelebAHQ
import argparse
from tqdm import tqdm


def main(resolution=256):
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    args = parser.parse_args()
    print(args)

    datas = MMCelebAHQ(
        root="~/data_hhd/mm-celeba-hq",
        size=resolution,
    )
    save_dir = f"assets/datasets/mmcelebahq{resolution}_features_withcaptioncontext/all"

    device = "cuda"
    os.makedirs(save_dir)

    autoencoder = libs.autoencoder.get_model(
        "assets/stable-diffusion/autoencoder_kl.pth"
    )
    autoencoder.to(device)
    clip = libs.clip.FrozenCLIPEmbedder()
    clip.eval()
    clip.to(device)

    with torch.no_grad():
        for idx, data in tqdm(enumerate(datas)):
            x, captions = data

            if len(x.shape) == 3:
                x = x[None, ...]
            x = torch.tensor(x, device=device)
            moments = autoencoder(x, fn="encode_moments").squeeze(0)
            moments = moments.detach().cpu().numpy()
            np.save(os.path.join(save_dir, f"{idx}.npy"), moments)

            latent = clip.encode(captions)
            for i in range(len(latent)):
                c = latent[i].detach().cpu().numpy()
                np.save(os.path.join(save_dir, f"{idx}_{i}.npy"), c)
                with open(os.path.join(save_dir, f"{idx}_{i}_captions.txt"), "w") as f:
                    for _caption in captions:
                        f.write(_caption)
                        f.write("\n")

            # if idx > 300:
            #    break


if __name__ == "__main__":
    main()
