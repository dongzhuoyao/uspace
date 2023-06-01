from configs.config_utils import update_config
from flow_matching import CNF
import ml_collections
import torch
from torch import multiprocessing as mp
import accelerate
import tools.utils_uvit as utils_uvit
from datasets import get_dataset
import tempfile
from absl import logging
import builtins
import libs.autoencoder
from torch.utils.data import DataLoader
import torchvision
import numpy as np
from absl import flags
from absl import app
from ml_collections import config_flags
import os

from tools.utils_vis import (
    extract_latents,
    sample_for_hspace_vis,
    extract_latents_and_attr,
)
from tools.utils_interp import cal_delta_change


def evaluate(config, vis_reversible=False):
    ###########################
    _exp_kwargs = config._exp_kwargs
    ############################
    mini_batch_size = config.sample.mini_batch_size

    if config.get("benchmark", False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method("spawn")
    accelerator = accelerate.Accelerator(
        # mixed_precision="fp16"
    )  # need True in save_latent
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f"Process {accelerator.process_index} using device: {device}")

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)
    if accelerator.is_main_process:
        utils_uvit.set_logger(log_level="info", fname=config.output_path)
    else:
        utils_uvit.set_logger(log_level="error")
        builtins.print = lambda *args: None

    dataset = get_dataset(**config.dataset)
    train_dataset = dataset.get_split(split="train", labeled=_exp_kwargs["has_attr"])
    train_dataset_loader = DataLoader(
        train_dataset,
        batch_size=mini_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=1,  # make it smaller for dissecting
        pin_memory=True,
        persistent_workers=True,
    )

    nnet = utils_uvit.get_nnet(**config.nnet)
    nnet = accelerator.prepare(nnet)
    logging.info(f"load nnet from {config.nnet_path}")
    accelerator.unwrap_model(nnet).load_state_dict(
        torch.load(config.nnet_path, map_location="cpu")
    )
    nnet.eval()

    autoencoder = libs.autoencoder.get_model(config.autoencoder.pretrained_path)
    autoencoder.to(device)

    @torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    def decode_large_batch(_batch):
        decode_mini_batch_size = 50  # use a small batch size since the decoder is large
        xs = []
        pt = 0
        for _decode_mini_batch_size in utils_uvit.amortize(
            _batch.size(0), decode_mini_batch_size
        ):
            x = decode(_batch[pt : pt + _decode_mini_batch_size])
            pt += _decode_mini_batch_size
            xs.append(x)
        xs = torch.concat(xs, dim=0)
        assert xs.size(0) == _batch.size(0)
        return xs

    def encode_large_batch(_batch):
        decode_mini_batch_size = 50  # use a small batch size since the decoder is large
        xs = []
        pt = 0
        for _decode_mini_batch_size in utils_uvit.amortize(
            _batch.size(0), decode_mini_batch_size
        ):
            x = encode(_batch[pt : pt + _decode_mini_batch_size])
            pt += _decode_mini_batch_size
            xs.append(x)
        xs = torch.concat(xs, dim=0)
        assert xs.size(0) == _batch.size(0)
        return xs

    score_model = CNF(net=nnet)

    logging.info(config.sample)
    assert os.path.exists(dataset.fid_stat)
    logging.info(
        f"sample: n_samples={config.sample.n_samples}, mode={config.train.mode}, mixed_precision={config.mixed_precision}"
    )

    _iter = iter(train_dataset_loader)

    def sample_fn(input_z, **kwargs):
        if config.train.mode == "uncond":
            _kwargs = dict(y=None)
        elif config.train.mode == "cond":
            _kwargs = dict(y=dataset.sample_label(len(input_z), device=device))
        else:
            raise NotImplementedError

        kwargs.update(_kwargs)

        _feat = score_model.decode(
            input_z,
            **kwargs,
        )
        return decode_large_batch(_feat)

    def encode_fn(**_kwargs):
        _real_data = next(_iter)
        kwargs = dict(y=None, **_kwargs)

        has_attr = kwargs["has_attr"]
        if has_attr:
            _real_data, _attr = _real_data
            kwargs.update(dict(attr=_attr))

        _real_data = _real_data.to("cuda")
        print("real_data", _real_data.shape, _real_data.dtype)
        _feat = (
            autoencoder.sample(_real_data)
            if "feature" in config.dataset.name
            else encode(_real_data)
        )

        _z_latent = score_model.encode(
            _feat,
            **kwargs,
        )

        if config.dissection.is_eval_vf_interp:
            _z_feat_recovered = score_model.decode(
                _z_latent,
                **kwargs,
            )
            cal_delta_change(_feat, _z_feat_recovered, config=config)
            print("only need one batch for calculating delta change, exit(0).")
            exit(0)

        if vis_reversible:  # For debugging
            _z_feat_recovered = score_model.decode(
                _z_latent,
                **kwargs,
            )
            logging.info(
                f"save image img-size,{_feat.shape}, {_z_feat_recovered.shape}, {torch.norm(_feat).item()}, {torch.norm(_feat - _z_feat_recovered).item()}"
            )

            if "feature" not in config.dataset.name:
                _data_recovered = decode(_z_feat_recovered)
                img_vis = torch.cat(
                    (
                        _real_data,
                        _data_recovered,
                    ),
                    dim=0,
                )
                img_vis = ((img_vis + 1) * 0.5).clamp(0, 1)
                torchvision.utils.save_image(
                    img_vis,
                    "reversible_vis.png",
                    padding=0,
                    pad_value=1.0,
                )

        if has_attr:
            _z_latent = (_z_latent, _attr)
        return _z_latent

    if config.dissection.is_eval_vf_interp:
        _exp_kwargs.write_scale = 1
        _npz = extract_latents(
            accelerator,
            encode_fn,
            **_exp_kwargs,
        )

    elif _exp_kwargs.dissect_name == "read":
        os.makedirs(config.sample.path, exist_ok=True)
        with tempfile.TemporaryDirectory() as temp_path:
            path = config.sample.path or temp_path
            if accelerator.is_main_process:
                os.makedirs(path, exist_ok=True)
            logging.info(f"Samples are saved in {path}")
            if _exp_kwargs["has_attr"]:
                _latent, _attr = extract_latents_and_attr(
                    accelerator,
                    encode_fn,
                    **_exp_kwargs,
                )
                logging.info(f"save latents to {_exp_kwargs.read_path_root}")

                np.savez(
                    os.path.join(_exp_kwargs.read_path_root, "latents.npy"),
                    latent=_latent,
                    attr=_attr,
                )
            else:
                _npz = extract_latents(
                    accelerator,
                    encode_fn,
                    **_exp_kwargs,
                )
                logging.info(f"save latents to {_exp_kwargs.read_path_root}")
                np.save(os.path.join(_exp_kwargs.read_path_root, "latents.npy"), _npz)

    elif _exp_kwargs.dissect_name in ["write_pca", "write_attr", "write_x0"]:
        os.makedirs(_exp_kwargs.vis_path, exist_ok=True)
        with tempfile.TemporaryDirectory() as temp_path:
            path = _exp_kwargs.vis_path or temp_path
            if accelerator.is_main_process:
                os.makedirs(path, exist_ok=True)
            logging.info(f"Samples are saved in {path}")
            sample_for_hspace_vis(
                accelerator,
                path,
                sample_fn,
                dataset.unpreprocess,
                z_shape=(config.z_shape[0], config.z_shape[1], config.z_shape[2]),
                device=device,
                **_exp_kwargs,
            )
    else:
        logging.info("skip")
        raise NotImplementedError


########
if False:  # CM_UNet
    cfg_path = "configs/lfm_cm256_unet_large.py"
    dissect_name = "write_attr"
    edit_loc = "head"

    write_path_root = "mid_feat_with_latentz_ssdstore/unet_realimg_celebamask256_features_cond_ep180000_head_n3000_euler100"


elif False:  # CM_UViT
    cfg_path = "configs/lfm_cm256_uvit_large.py"
    dissect_name = "read"
    edit_loc = "write_attr"
    write_path_root = "mid_feat_with_latentz_ssdstore/uvit_realimg_celebamask256_features_cond_ep110000_head_n10000_euler100"

elif True:  # CM_UViT
    cfg_path = "configs/lfm_cm256_uvit_large.py"
    dissect_name = "write_attr"
    edit_loc = "tail"
    write_path_root = "mid_feat_with_latentz_ssdstore/uvit_realimg_celebamask256_features_cond_ep110000_euler_step0.01-dopri5_tail_n5000"


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", cfg_path, "Training configuration.", lock_config=False
)
flags.DEFINE_string("exp", None, "experiment to do.")  # extract_latents
flags.DEFINE_string("output_path", None, "The path to output log.")


def main(argv):
    config = FLAGS.config
    config.output_path = FLAGS.output_path

    ###################
    config.nnet_path = config.dissection.ckpt_path_to_dissect
    config.dissection.seed = config.seed
    config.dissection.write_path_root = write_path_root
    config.dissection.edit_loc = edit_loc
    logging.warning(f"seed {config.seed}")

    config.sample.n_samples = config.dissection.n_samples
    config.sample.mini_batch_size = config.dissection.mini_batch_size
    config.dissection.dissect_name = dissect_name

    config._exp_kwargs = config.dissection

    config = update_config(config)  # final step
    ###################

    evaluate(config)


if __name__ == "__main__":
    app.run(main)
