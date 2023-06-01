import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import os
from tqdm import tqdm
from absl import logging


# list the attribute of celeba dataset in order
# 0: 5_o_Clock_Shadow
#
# https://raw.githubusercontent.com/taki0112/StarGAN-Tensorflow/master/dataset/celebA/list_attr_celeba.txt
CelebA_ATTR40 = (  # https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image/celeba.py
    "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs "
    "Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair "
    "Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair "
    "Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache "
    "Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline "
    "Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings "
    "Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
).split()

"""
0 5_o_Clock_Shadow
1 Arched_Eyebrows
2 Attractive
3 Bags_Under_Eyes
4 Bald
5 Bangs
6 Big_Lips
7 Big_Nose
8 Black_Hair
9 Blond_Hair
10 Blurry
11 Brown_Hair
12 Bushy_Eyebrows
13 Chubby
14 Double_Chin
15 Eyeglasses
16 Goatee
17 Gray_Hair
18 Heavy_Makeup
19 High_Cheekbones
20 Male
21 Mouth_Slightly_Open
22 Mustache
23 Narrow_Eyes
24 No_Beard
25 Oval_Face
26 Pale_Skin
27 Pointy_Nose
28 Receding_Hairline
29 Rosy_Cheeks
30 Sideburns
31 Smiling
32 Straight_Hair
33 Wavy_Hair
34 Wearing_Earrings
35 Wearing_Hat
36 Wearing_Lipstick
37 Wearing_Necklace
38 Wearing_Necktie
39 Young
"""

# 4, bald
# 7, big nose
# 20, Male
# 15, Eyeglasses
# 31, Smiling
# 22, Mustache

# 39, young


FFHQ_ATTR11 = [
    "gender",
    "smile",
    "no_glasses",
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
]


def should_ignore(_name):
    if _name.startswith("pca"):
        return True
    elif _name.startswith("latent"):
        return True
    elif _name.startswith("delta"):
        return True
    else:
        return False


def _attr_name_from_attr_id(ith_attr, dataset_name):
    if "ffhq" in dataset_name:
        return FFHQ_ATTR11[ith_attr]
    elif "celeba" in dataset_name:  # CelebA, CelebAMask-HQ
        return CelebA_ATTR40[ith_attr]
    else:
        raise ValueError("unknown dataset_name", dataset_name)


def get_attr_name_from_attr_id(ith_attr, dataset_name):
    if isinstance(ith_attr, int):
        return _attr_name_from_attr_id(ith_attr, dataset_name)
    elif isinstance(ith_attr, str):
        ith_attrs = [int(tmp) for tmp in ith_attr.split("_")]
        attr_names = [_attr_name_from_attr_id(_id, dataset_name) for _id in ith_attrs]
        return "_".join(attr_names)
    else:
        raise ValueError("unknown ith_attr", ith_attr)


def cal_delta_direction(attr_id, attrs, feats):
    if attrs.shape[1] == 40:
        _attr_name = CelebA_ATTR40[attr_id]
    elif attrs.shape[1] == 11:
        _attr_name = FFHQ_ATTR11[attr_id]
    else:
        raise ValueError("unknown attr dim", len(attrs))

    _attr = attrs[:, attr_id]  # [B,]
    _pos_feat = feats[_attr == 1]  # [B, T, C, W, H]
    _neg_feat = feats[_attr == 0]  # [B, T, C, W, H]
    print(
        f"attr_id={attr_id}, {_attr_name}, total_len: {len(_attr)}, _pos_feat.shape={_pos_feat.shape}, _neg_feat.shape={_neg_feat.shape}"
    )
    _pos_feat = np.mean(_pos_feat, axis=0, keepdims=True)  # [1, T, C, W, H]
    _neg_feat = np.mean(_neg_feat, axis=0, keepdims=True)  # [1, T, C, W, H]
    _delta_feat = _pos_feat - _neg_feat  # [1, T, C, W, H]

    return _delta_feat


def cal_latentz_delta(read_path_root, latent_file, attr_dim, is_debug=False):
    _data = np.load(os.path.join(read_path_root, latent_file))
    _attrs, _latent = _data["attr"], _data["latent"]

    cal_delta_direction(0, _attrs, _latent)
    _delta_feat_list = [
        cal_delta_direction(_attr_id, _attrs, _latent)
        for _attr_id in tqdm(range(attr_dim), desc="attr_id", total=attr_dim)
    ]
    _delta_feat_list = np.concatenate(_delta_feat_list, axis=0)  # [attr_dim, C, W, H]
    _target_npy_path = f"{read_path_root}/delta_latentz"
    _target_npy_path = _target_npy_path + "_debug" if is_debug else _target_npy_path
    np.save(_target_npy_path, _delta_feat_list)


def extract_hspace_feat_unet_by_attr(
    read_path_root,
    batch_num,
    latent_file="latents.npy.npz",
    is_debug=False,
    cal_latentz_delta_only=False,
):
    _attrs = np.load(os.path.join(read_path_root, latent_file))["attr"]  # [B, attr_dim]
    attr_dim = _attrs.shape[1]
    if cal_latentz_delta_only:
        cal_latentz_delta(read_path_root, latent_file, attr_dim)
        return
    names = os.listdir(read_path_root)
    names = [_name for _name in names if not should_ignore(_name)]
    timesteps_str = set([_name.split("_")[1].replace(".npy", "") for _name in names])

    print("timesteps_str", timesteps_str)
    print("***")
    print("time step", len(timesteps_str))
    _all_feat = []
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
        _all_feat.append(np.expand_dims(_feat_concat_np, axis=1))
    _all_feat = np.concatenate(_all_feat, axis=1)  # [B,T,C,W,H]

    _delta_feat_list = [
        cal_delta_direction(_attr_id, _attrs, _all_feat)
        for _attr_id in tqdm(range(attr_dim), desc="attr_id", total=attr_dim)
    ]
    _delta_feat_list = np.concatenate(
        _delta_feat_list, axis=0
    )  # [attr_dim, T, C, W, H]

    for _tid, _timestep in enumerate(timesteps_str):
        _target_npy_path = f"{read_path_root}/delta_{_timestep}"
        if is_debug:
            _target_npy_path = _target_npy_path + "_debug"
        np.save(_target_npy_path, _delta_feat_list[:, _tid])
    print("batch num", batch_num)


if __name__ == "__main__":
    if False:
        extract_hspace_feat_unet_by_attr(
            read_path_root="mid_feat_with_latentz/unet_realimg_ffhq256_features_latent0_n20000_euler100",
            batch_num=200,
            latent_file="latents.npy",
            is_debug=False,
        )
    if False:
        extract_hspace_feat_unet_by_attr(
            read_path_root="mid_feat_with_latentz_ssdstore/unet_realimg_celebamask256_features_cond_ep80000_latent0_n1000_euler100",
            batch_num=10,
            is_debug=False,
            cal_latentz_delta_only=True,
        )
        #
    if False:
        extract_hspace_feat_unet_by_attr(
            read_path_root="mid_feat_with_latentz_ssdstore/unet_realimg_celebamask256_features_cond_ep180000_head_n3000_euler100",
            batch_num=30,
            is_debug=False,
            cal_latentz_delta_only=False,
        )
    if False:  # main func
        extract_hspace_feat_unet_by_attr(
            read_path_root="mid_feat_with_latentz_ssdstore/uvit_realimg_celebamask256_features_cond_ep110000_head_n10000_euler100",
            batch_num=20,
            is_debug=False,
            cal_latentz_delta_only=False,
        )
    if True:  # new for mid
        extract_hspace_feat_unet_by_attr(
            read_path_root="mid_feat_with_latentz_ssdstore/uvit_realimg_celebamask256_features_cond_ep110000_euler_step0.01-dopri5_mid_n5000",
            batch_num=2,
            is_debug=False,
            cal_latentz_delta_only=False,
        )
    if False:  # new for tail
        extract_hspace_feat_unet_by_attr(
            read_path_root="mid_feat_with_latentz_ssdstore/uvit_realimg_celebamask256_features_cond_ep110000_euler_step0.01-dopri5_tail_n5000",
            batch_num=10,
            is_debug=False,
            cal_latentz_delta_only=False,
        )
