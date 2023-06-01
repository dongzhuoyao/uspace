from tqdm import tqdm


def rename_files_in_dir(
    path="mid_feat_with_latentz_ssdstore/uvit_realimg_celebamask256_features_cond_ep110000_euler_step0.01-dopri5_tail_n5000",
    prefix="tail_",
):
    import os
    import glob

    for _f in tqdm(glob.glob(os.path.join(path, "*"))):
        _basename = os.path.basename(_f)
        if _basename.startswith(prefix):
            _basename = _basename.replace(prefix, "")
            _f_new = os.path.join(path, _basename)
            print("a")
            os.rename(_f, _f_new)


if __name__ == "__main__":
    rename_files_in_dir()
