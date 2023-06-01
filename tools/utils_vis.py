import os
import time
import torch
import numpy as np
from sklearn.decomposition import PCA
from einops import rearrange

from tools.utils_attr import get_attr_name_from_attr_id
from tools.utils_uvit import amortize
from tqdm import tqdm
import torchvision
from torchvision.io import read_image
from torchvision.utils import make_grid
import time
from absl import logging
import itertools

_PADDING = 8


def pretty_datetime():
    return time.strftime("%m-%d_%H-%M-%S", time.gmtime(time.time()))


def load_z_from_dir(root_path, has_attr, device):
    print("*" * 44)
    print("load z from dir: ", root_path)
    if has_attr:
        z = np.load(root_path + ".npz")["latent"]
    else:
        z = np.load(root_path)

    print("z shape: ", z.shape)
    print("*" * 44)
    return torch.from_numpy(z).to(device)


def get_1d_pca_components_sklearn(
    mt,
    svd_solver="full",
    n_components=0.9,
):
    """
    mt: [N, C, H, W], N is the number of samples, C is the number of channels, torch tensor
    return: [n_components, C, H, W], torch tensor
    """

    logging.info(f"running run_pca_sklearn, mt.shape={mt.shape}")

    assert len(mt.shape) == 2

    mt = mt.cpu().numpy().astype("float32")

    # using full dimension as component
    pca = PCA(n_components=n_components, svd_solver=svd_solver)
    pca.fit(mt)

    _components = torch.from_numpy(pca.components_)
    return _components


def get_1d_pca_components_faiss(mt, n_components):
    import faiss

    print("faiss_pca, mt shape: ", mt.shape)
    bs, _feat_dim = mt.shape
    # mt = np.random.rand(1000, _feat_dim).astype("float32")
    mt = mt.cpu().numpy().astype("float32")
    mat = faiss.PCAMatrix(_feat_dim, n_components)
    mat.train(mt)
    assert mat.is_trained
    _components = faiss.vector_to_array(mat.A).reshape(
        mat.d_out, mat.d_in
    )  # https://gist.github.com/mdouze/ca65bce66f77cd2ef4df8769e19443a9
    print(_components.shape)

    return torch.from_numpy(_components)


def get_pca_components_sklearn(
    mt,
    svd_solver="full",
    n_components=0.9,
    is_check_orthogonality=True,
):  # https://github.com/harskish/ganspace/blob/master/estimators.py#L84
    """
    mt: [N, C, H, W], N is the number of samples, C is the number of channels, torch tensor
    return: [n_components, C, H, W], torch tensor
    """

    logging.info(f"running run_pca_sklearn, mt.shape={mt.shape}")

    assert (
        len(mt.shape) >= 3
    ), "mt should be [N, C, H, W] or [N,C,T], not mt.shape={}".format(mt.shape)

    _shape = mt.shape[1:]

    mt = mt.reshape(len(mt), -1).cpu().numpy().astype("float32")

    # using full dimension as component
    pca = PCA(n_components=n_components, svd_solver=svd_solver)

    pca.fit(mt)

    _components = pca.components_
    if is_check_orthogonality:
        # Check orthogonality
        dotps = [
            np.dot(_components[i], _components[j])
            for (i, j) in itertools.combinations(range(len(_components)), 2)
        ]
        if not np.allclose(dotps, 0, atol=1e-4):
            print("IPCA components not orghogonal, max dot", np.abs(dotps).max())

    _components = torch.from_numpy(_components.reshape(-1, *_shape))
    return _components


def move_delta_z(start, delta, scales):
    """
    start: [B, C, W, H]
    delta: [C, W, H]
    scales: [S]
    return: [B, S, C, W, H]
    """
    if isinstance(scales, list) or isinstance(scales, tuple):
        scales = np.array(scales)
    assert isinstance(scales, np.ndarray), scales
    logging.info(f"Move delta z:  scale: {scales}")
    scales = torch.from_numpy(scales).to(start.device)
    scales = rearrange(scales, "S -> 1 S 1 1 1")
    start = rearrange(start, "B C W H-> B 1 C W H")
    delta = rearrange(delta, "1 C W H-> 1 1 C W H")
    return start + delta * scales


def sample_for_hspace_vis(
    accelerator,
    path,
    sample_fn,
    unpreprocess_fn=None,
    padding=_PADDING,
    pad_value=1.0,
    z_shape=None,
    device=None,
    n_samples=None,
    mini_batch_size=None,
    write_scales=None,
    fixed_z_path=None,
    **kwargs,
):
    os.makedirs(path, exist_ok=True)
    idx = 0
    batch_size = mini_batch_size * accelerator.num_processes
    _seed = kwargs.get("seed", None)

    if (
        kwargs["dissect_name"] in ["write_pca", "write_attr", "write_x0"]
        and fixed_z_path is not None
    ):
        _latent_z = load_z_from_dir(
            fixed_z_path, has_attr=kwargs["has_attr"], device=device
        )
        n_samples = len(_latent_z)
        logging.warning(f"fixed_z_path is None, override n_samples: {n_samples}")

    for _batch_id, _batch_size in enumerate(
        tqdm(
            amortize(n_samples, batch_size),
            disable=not accelerator.is_main_process,
            desc="sample_for_hspace_vis",
        )
    ):
        if kwargs["dissect_name"] == "read":
            input_z = torch.randn(mini_batch_size, *z_shape, device=device)
            samples = unpreprocess_fn(
                sample_fn(input_z=input_z, batch_id=_batch_id, **kwargs)
            )
        elif kwargs["dissect_name"] in ["write_pca", "write_attr"]:
            samples_concat = []
            if fixed_z_path is None:
                input_z = torch.randn(mini_batch_size, *z_shape, device=device)
            else:
                input_z = _latent_z[
                    _batch_id * mini_batch_size : (_batch_id + 1) * mini_batch_size
                ]

            for write_scale in write_scales:
                logging.info(f"write scale: {write_scale}")
                samples = unpreprocess_fn(
                    sample_fn(
                        input_z=input_z,
                        write_scale=write_scale,
                        batch_id=_batch_id,
                        **kwargs,
                    )
                )
                samples_concat.append(rearrange(samples, "b c h w -> b 1 c h w"))
            samples = torch.cat(samples_concat, dim=1)
            samples = rearrange(samples, "b s c h w -> (b s) c h w")

        elif kwargs["dissect_name"] in ["write_x0"]:
            samples_concat = []
            if fixed_z_path is None:
                input_z = torch.randn(mini_batch_size, *z_shape, device=device)
            else:
                input_z = _latent_z[
                    _batch_id * mini_batch_size : (_batch_id + 1) * mini_batch_size
                ]

            _direction = np.load(
                os.path.join(kwargs["write_path_root"], "delta_latentz.npy")
            )[kwargs.get("ith_attr", None)]
            _direction = rearrange(
                torch.from_numpy(_direction).to(device), "c w h -> 1 c w h"
            )

            for write_scale in write_scales:
                logging.info(f"write scale: {write_scale}")
                input_z_changed = input_z + write_scale * _direction
                samples = unpreprocess_fn(
                    sample_fn(
                        input_z=input_z_changed,
                        write_scale=write_scale,
                        batch_id=_batch_id,
                        **kwargs,
                    )
                )
                samples_concat.append(rearrange(samples, "b c h w -> b 1 c h w"))
            samples = torch.cat(samples_concat, dim=1)
            samples = rearrange(samples, "b s c h w -> (b s) c h w")

        else:
            raise NotImplementedError(
                f"dissect_name should be read or write, but got: {kwargs['dissect_name']}"
            )

        ith_attr = kwargs.get("ith_attr", None)
        _attr_name = get_attr_name_from_attr_id(ith_attr, kwargs["dataset_name"])
        samples = accelerator.gather(samples.contiguous())  # [:_batch_size]
        if accelerator.is_main_process:
            scales = "_".join([f"{s:.2f}" for s in write_scales])
            img_path = os.path.join(
                path,
                f"{pretty_datetime()}_seed{_seed}_{idx}_{_attr_name}{scales}.png",
            )

            grid = make_grid(
                samples, nrow=len(write_scales), padding=padding, pad_value=pad_value
            )
            img = torchvision.transforms.ToPILImage()(grid)

            img.save(img_path)
            logging.info(f"save image to {img_path}")
        idx += 1


def sample_for_t2i_vis(
    accelerator,
    path,
    sample_fn,
    unpreprocess_fn=None,
    padding=_PADDING,
    pad_value=1.0,
    z_shape=None,
    device=None,
    n_samples=None,
    mini_batch_size=None,
    fixed_z_path=None,
    **kwargs,
):
    os.makedirs(path, exist_ok=True)
    idx = 0
    batch_size = mini_batch_size * accelerator.num_processes
    _seed = kwargs.get("seed", None)

    if fixed_z_path is not None:
        _latent_z = load_z_from_dir(
            fixed_z_path, has_attr=kwargs["has_attr"], device=device
        )
        n_samples = len(_latent_z)
        logging.warning(f"fixed_z_path is None, override n_samples: {n_samples}")

    for _batch_id, _batch_size in enumerate(
        tqdm(
            amortize(n_samples, batch_size),
            disable=not accelerator.is_main_process,
            desc="sample_for_t2i_vis",
        )
    ):
        if kwargs["dissect_name"] == "read":
            input_z = torch.randn(mini_batch_size, *z_shape, device=device)
            samples = unpreprocess_fn(
                sample_fn(input_z=input_z, batch_id=_batch_id, **kwargs)
            )
        elif kwargs["dissect_name"] in ["write_pca", "write_attr"]:
            if fixed_z_path is None:
                input_z = torch.randn(mini_batch_size, *z_shape, device=device)
            else:
                input_z = _latent_z[
                    _batch_id * mini_batch_size : (_batch_id + 1) * mini_batch_size
                ]

            samples = unpreprocess_fn(
                sample_fn(
                    input_z=input_z,
                    batch_id=_batch_id,
                    **kwargs,
                )
            )

        else:
            raise NotImplementedError(
                f"dissect_name should be read or write, but got: {kwargs['dissect_name']}"
            )

        ith_attr = kwargs.get("ith_attr", None)
        _attr_name = get_attr_name_from_attr_id(ith_attr, kwargs["dataset_name"])
        samples = accelerator.gather(samples.contiguous())  # [:_batch_size]
        if accelerator.is_main_process:
            scales = "_".join([f"{s:.2f}" for s in write_scales])
            img_path = os.path.join(
                path,
                f"{pretty_datetime()}_seed{_seed}_{idx}_{_attr_name}{scales}.png",
            )

            grid = make_grid(
                samples, nrow=len(write_scales), padding=padding, pad_value=pad_value
            )
            img = torchvision.transforms.ToPILImage()(grid)

            img.save(img_path)
            logging.info(f"save image to {img_path}")
        idx += 1


def sample_for_scalevis(
    accelerator,
    path,
    n_samples,
    mini_batch_size,
    sample_fn,
    unpreprocess_fn=None,
    scales_n=None,
    padding=_PADDING,
    pad_value=1.0,
):
    os.makedirs(path, exist_ok=True)
    idx = 0
    batch_size = mini_batch_size * accelerator.num_processes

    for _batch_size in tqdm(
        amortize(n_samples, batch_size),
        disable=not accelerator.is_main_process,
        desc="sample_for_scalevis",
    ):
        samples = unpreprocess_fn(sample_fn(mini_batch_size))
        samples = accelerator.gather(samples.contiguous())  # [:_batch_size]
        if accelerator.is_main_process:
            grid = make_grid(
                samples, nrow=scales_n, padding=padding, pad_value=pad_value
            )
            img = torchvision.transforms.ToPILImage()(grid)
            img.save(os.path.join(path, f"{idx}.png"))
            logging.info("save image to {}".format(os.path.join(path, f"{idx}.png")))
        idx += 1


def extract_latents(
    accelerator,
    encode_fn,
    n_samples,
    mini_batch_size,
    **kwargs,
):
    idx = 0
    batch_size = mini_batch_size
    npz_all = []
    for _batch_id, _batch_size in tqdm(
        enumerate(amortize(n_samples, batch_size)),
        disable=False,
        desc="sample latents to npz",
    ):
        _latents = encode_fn(batch_id=_batch_id, **kwargs)
        _latents = _latents.contiguous()[:_batch_size]
        npz_all.append(_latents)
        idx += 1

    npz_all = torch.cat(npz_all, 0).cpu().numpy()
    return npz_all


def extract_latents_and_attr(
    accelerator,
    sample_fn,
    n_samples,
    mini_batch_size,
    **kwargs,
):
    idx = 0
    batch_size = mini_batch_size
    _feat_list, _attr_list = [], []
    for _batch_id, _batch_size in enumerate(tqdm(amortize(n_samples, batch_size))):
        _feat, _attrs = sample_fn(batch_id=_batch_id, **kwargs)
        _feat = _feat.contiguous()[:_batch_size]
        _attrs = _attrs.contiguous()[:_batch_size]

        _feat_list.append(_feat)
        _attr_list.append(_attrs)

        idx += 1

    _feat_list = torch.cat(_feat_list, 0).cpu().numpy()
    _attr_list = torch.cat(_attr_list, 0).cpu().numpy()

    return _feat_list, _attr_list


if __name__ == "__main__":
    if True:
        pass
