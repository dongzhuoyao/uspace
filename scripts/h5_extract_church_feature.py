import sys,os
path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)

from lfm_dataset.lsun import LSUNChurchesTrain

import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader
from libs.autoencoder import get_model
import argparse
import h5py
from tqdm import tqdm
torch.manual_seed(0)
np.random.seed(0)





def main(resolution=256, debug=False, ds_name='church',
        _cluster_h5_path='assets/datasets/church256_features.h5'):
    parser = argparse.ArgumentParser()
    

    dataset = LSUNChurchesTrain()
    
    
    train_dataset_loader = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False,
                                      num_workers=8, pin_memory=True, persistent_workers=True)

    model = get_model('assets/stable-diffusion/autoencoder_kl.pth')
    model = nn.DataParallel(model)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)


    if debug:
        _cluster_h5_path = _cluster_h5_path.replace(".h5", "debug.h5")

    _len = len(train_dataset_loader.dataset)
    print('create dataset, length: ', _len)

    f = h5py.File(_cluster_h5_path, mode="w")
    f.close()

    f = h5py.File(_cluster_h5_path, mode="a")
    f.create_dataset(
        "train_feat", data=np.ones(shape=(_len, 8, 32, 32), dtype=np.float32) * -1
    )
    

    dset = f.create_dataset("all_attributes", (1,))
    dset.attrs["dataset_name"] = ds_name



    idx = 0
    for batch in tqdm(train_dataset_loader, desc='{}/{}'.format(idx, _len)):
        img, label = batch
        #img = torch.cat([img, img.flip(dims=[-1])], dim=0)
        #label = torch.cat([label, label], dim=0)

        img = img.to(device)
        moments = model(img, fn='encode_moments')
        moments = moments.detach().cpu().numpy()

        #label = label.detach().cpu().numpy()

        for moment, lb in zip(moments, label):
            f["train_feat"][idx, :] = moment
            idx += 1

    print(f'save {idx} files')
    f.close()



if __name__ == "__main__":
    main()
