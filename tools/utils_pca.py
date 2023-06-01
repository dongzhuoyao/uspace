import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange
import os
from tools.utils_attr import should_ignore
from tools.utils_vis import get_1d_pca_components_faiss, get_1d_pca_components_sklearn
from tqdm import tqdm


def extract_hspace_feat_unet_by_pca(
    read_path_root="mid_feat/unet_latent0_euler100",
    n_components=50,
    batch_num=10,
    is_debug=False,
):
    names = os.listdir(read_path_root)
    names = [_name for _name in names if not should_ignore(_name)]
    timesteps_str = set([_name.split("_")[1].replace(".npy", "") for _name in names])

    print("timesteps_str", timesteps_str)
    print("***")
    print("time step", len(timesteps_str))
    for _timestep in tqdm(timesteps_str, desc="timestep"):
        _target_npy_path = f"{read_path_root}/pca{n_components}_{_timestep}"
        if is_debug:
            _target_npy_path = _target_npy_path + "_debug"

        _feat_concat = []
        for _batch_id in range(batch_num):
            name = f"{_batch_id}_{_timestep}.npy"
            _feat = np.load(os.path.join(read_path_root, name))
            _feat_concat.append(_feat)

        if len(_feat_concat) == 0:
            print("**** empty feat", name)
            raise ValueError("**** empty feat", name)
        _feat_concat_np = np.concatenate(_feat_concat, axis=0)  # [B,C,W,H]
        _feat_concat_np = torch.from_numpy(_feat_concat_np)
        b, c, w, h = _feat_concat_np.shape
        _feat_concat_np = rearrange(_feat_concat_np, "b c w h -> b (c w h)")
        print("pca, _feat_concat.shape", _feat_concat_np.shape)
        _components = get_1d_pca_components_faiss(
            _feat_concat_np, n_components=n_components
        )
        _components = rearrange(_components, "b (c w h) -> b c w h", c=c, w=w, h=h)
        np.save(_target_npy_path, _components.cpu().numpy())
        print("save", _target_npy_path)


def extract_hspace_feat_unet_by_pca_b_tc(
    read_path_root="mid_feat/unet_euler100", n_components=5, batch_num=1, is_debug=False
):
    names = os.listdir(read_path_root)
    names = [_name for _name in names if not should_ignore(_name)]
    timesteps_str = set([_name.split("_")[1].replace(".npy", "") for _name in names])

    print("timesteps_str", timesteps_str)
    print("***")
    print("time step", len(timesteps_str))
    result_b_time_c = []
    for _timestep in tqdm(timesteps_str, desc="timestep"):
        _feat_concat = []
        for _batch_id in range(batch_num):
            name = f"{_batch_id}_{_timestep}.npy"
            _feat = np.load(os.path.join(read_path_root, name))
            _feat_concat.append(_feat)

        if len(_feat_concat) == 0:
            print("**** empty feat", name)
            raise ValueError("**** empty feat", name)
        _feat_concat_np = np.concatenate(_feat_concat, axis=0)  # [B,C,W,H]
        result_b_time_c.append(np.expand_dims(_feat_concat_np, axis=1))  # [B,1,C,W,H]
    result_b_time_c = np.concatenate(result_b_time_c, axis=1)  # [B,T,C,W,H]

    result_b_time_c = torch.from_numpy(result_b_time_c)
    b, t, c, w, h = result_b_time_c.shape
    result_b_time_c = rearrange(result_b_time_c, "b t c w h -> b (t c w h)")

    print("pca, result_b_time_c.shape", result_b_time_c.shape)
    _components = get_1d_pca_components_sklearn(
        result_b_time_c, n_components=n_components
    )

    _components = rearrange(
        _components, "b (t c w h) -> b t c w h", t=t, c=c, w=w, h=h
    )  # [batch_size, 6553600]?, 6553600=100*1024*8*8
    for _tid, _timestep in enumerate(timesteps_str):
        _target_npy_path = f"{read_path_root}/pca{n_components}_b_tc_{_timestep}"
        if is_debug:
            _target_npy_path = _target_npy_path + "_debug"
        np.save(_target_npy_path, _components[:, _tid, :].cpu().numpy())


if __name__ == "__main__":
    if False:  # main
        extract_hspace_feat_unet_by_pca(
            read_path_root="mid_feat_with_latentz_ssdstore/uvit_realimg_celebamask256_features_cond_ep110000_head_n10000_euler100",
            batch_num=20,
            n_components=100,
            is_debug=False,
        )

    if True:  # new for mid
        extract_hspace_feat_unet_by_pca(
            read_path_root="mid_feat_with_latentz_ssdstore/uvit_realimg_celebamask256_features_cond_ep110000_euler_step0.01-dopri5_mid_n5000",
            batch_num=10,
            is_debug=False,
            n_components=100,
        )
    if False:  # new for tail
        extract_hspace_feat_unet_by_pca(
            read_path_root="mid_feat_with_latentz_ssdstore/uvit_realimg_celebamask256_features_cond_ep110000_euler_step0.01-dopri5_tail_n5000",
            batch_num=10,
            is_debug=False,
            n_components=100,
        )
