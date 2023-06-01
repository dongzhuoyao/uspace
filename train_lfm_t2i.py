import ml_collections
import torch
from torch import multiprocessing as mp
from datasets import get_dataset
from torchvision.utils import make_grid, save_image
import tools.utils_uvit as utils_uvit
import einops
from torch.utils._pytree import tree_map
import accelerate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import tempfile
from tools.fid_score import calculate_fid_given_paths
from absl import logging
import builtins
import os
import wandb
import libs.autoencoder
import numpy as np
from flow_matching_t2i import CNF
from absl import flags
from absl import app
from ml_collections import config_flags
import sys
from pathlib import Path


def train(config):
    if config.get("benchmark", False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method("spawn")
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f"Process {accelerator.process_index} using device: {device}")

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)

    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.init(
            dir=os.path.abspath(config.workdir),
            project="lfm_uvit",
            config=config.to_dict(),
            name=config.hparams,
            job_type="train",
            mode="online",
        )
        utils_uvit.set_logger(
            log_level="info", fname=os.path.join(config.workdir, "output.log")
        )
        logging.info(config)
    else:
        utils_uvit.set_logger(log_level="error")
        builtins.print = lambda *args: None
    logging.info(f"Run on {accelerator.num_processes} devices")

    dataset = get_dataset(**config.dataset)
    assert os.path.exists(dataset.fid_stat)
    train_dataset = dataset.get_split(split="train", labeled=True)
    train_dataset_loader = DataLoader(
        train_dataset,
        batch_size=mini_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    test_dataset = dataset.get_split(split="test", labeled=True)  # for sampling
    test_dataset_loader = DataLoader(
        test_dataset,
        batch_size=config.sample.mini_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    train_state = utils_uvit.initialize_train_state(config, device)
    (
        nnet,
        nnet_ema,
        optimizer,
        train_dataset_loader,
        test_dataset_loader,
    ) = accelerator.prepare(
        train_state.nnet,
        train_state.nnet_ema,
        train_state.optimizer,
        train_dataset_loader,
        test_dataset_loader,
    )
    lr_scheduler = train_state.lr_scheduler

    # train_state.resume(config.ckpt_root)

    if len(os.listdir(config.ckpt_root)):
        logging.warning(
            "ckpt_root is True, will load[resume] pretrained model from {}".format(
                config.ckpt_root
            )
        )
        train_state.resume(config.ckpt_root)
    else:
        if config.pretrained_path is not None:
            logging.warning(
                "pretrained_path is True, will load pretrained model from {}".format(
                    config.pretrained_path
                )
            )

            if config.nnet.name == "unet_t2i":
                train_state.load_sd_unet_remove_attention(
                    config.pretrained_path, is_strict=False
                )
            elif config.nnet.name == "uvit_t2i":
                train_state.load_nnet_mscoco_or_in256(config.pretrained_path)
            else:
                raise not NotImplementedError
        else:
            logging.warning("pretrained_path is None, will train from scratch")

    autoencoder = libs.autoencoder.get_model(**config.autoencoder)
    autoencoder.to(device)

    @torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    def get_data_generator():
        while True:
            for data in tqdm(
                train_dataset_loader,
                disable=not accelerator.is_main_process,
                desc="epoch",
            ):
                yield data

    data_generator = get_data_generator()

    def get_context_generator():
        while True:
            for data in test_dataset_loader:
                _, _context = data
                yield _context

    context_generator = get_context_generator()

    # set the score_model to train
    score_model = CNF(net=nnet)
    score_model_ema = CNF(net=nnet_ema)

    def get_fixed_noise(batch_size, device, sample_channels, sample_resolution):
        fixed_noise = torch.randn(
            (
                batch_size * torch.cuda.device_count(),
                sample_channels,
                sample_resolution,
                sample_resolution,
            ),
            device=device,
        )
        return fixed_noise

    fixed_noise = get_fixed_noise(
        batch_size=config.vis_num,
        device=device,
        sample_channels=config.z_shape[0],
        sample_resolution=config.z_shape[1],
    )

    def train_step(_batch):
        _metrics = dict()
        optimizer.zero_grad()
        _z = (
            autoencoder.sample(_batch[0])
            if "feature" in config.dataset.name
            else encode(_batch[0])
        )

        loss = score_model.training_losses(
            _z,
            context=_batch[1],
            sigma_min=config.dynamic.sigma_min,
            **config.dissection,
        )
        _metrics["loss"] = accelerator.gather(loss.detach()).mean()
        accelerator.backward(loss.mean())
        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get("ema_rate", 0.9999))
        train_state.step += 1
        return dict(lr=train_state.optimizer.param_groups[0]["lr"], **_metrics)

    def dpm_solver_sample(_n_samples, _sample_steps, context):
        _vis_num = min(len(fixed_noise), len(context), config.vis_num)
        _z = score_model.decode(
            fixed_noise[:_vis_num],
            context=context[:_vis_num],
            **config.dissection,
        )
        return decode(_z)

    def eval_step(n_samples, sample_steps):
        logging.info(
            f"eval_step: n_samples={n_samples}, sample_steps={sample_steps},"
            f"mini_batch_size={config.sample.mini_batch_size}"
        )

        def sample_fn(_n_samples):
            _context = next(context_generator)
            assert len(_context) == _n_samples
            return dpm_solver_sample(_n_samples, sample_steps, context=_context)

        with tempfile.TemporaryDirectory() as temp_path:
            path = config.sample.path or temp_path
            if accelerator.is_main_process:
                os.makedirs(path, exist_ok=True)
            utils_uvit.sample2dir(
                accelerator,
                path,
                n_samples,
                config.sample.mini_batch_size,
                sample_fn,
                dataset.unpreprocess,
            )

            _fid = 0
            if accelerator.is_main_process:
                _fid = calculate_fid_given_paths((dataset.fid_stat, path))
                logging.info(f"step={train_state.step} fid{n_samples}={_fid}")
                with open(os.path.join(config.workdir, "eval.log"), "a") as f:
                    print(f"step={train_state.step} fid{n_samples}={_fid}", file=f)
                wandb.log({f"fid{n_samples}": _fid}, step=train_state.step)
            _fid = torch.tensor(_fid, device=device)
            _fid = accelerator.reduce(_fid, reduction="sum")

        return _fid.item()

    logging.info(
        f"Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}"
    )

    step_fid = []
    while train_state.step < config.train.n_steps:
        nnet.train()
        batch = tree_map(lambda x: x.to(device), next(data_generator))
        metrics = train_step(batch)

        nnet.eval()
        if (
            accelerator.is_main_process
            and train_state.step % config.train.log_interval == 0
        ):
            logging.info(utils_uvit.dct2str(dict(step=train_state.step, **metrics)))
            logging.info(config.workdir)
            wandb.log(metrics, step=train_state.step)

        if (
            accelerator.is_main_process
            and train_state.step % config.train.eval_interval == 0
        ):
            torch.cuda.empty_cache()
            logging.info("Save a grid of images...")
            contexts = torch.tensor(dataset.contexts, device=device)[: 2 * 5]
            samples = dpm_solver_sample(
                _n_samples=2 * 5, _sample_steps=50, context=contexts
            )
            samples = make_grid(dataset.unpreprocess(samples), 5)
            save_image(
                samples, os.path.join(config.sample_dir, f"{train_state.step}.png")
            )
            wandb.log({"samples": wandb.Image(samples)}, step=train_state.step)
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

        if (
            train_state.step % config.train.save_interval == 0
            or train_state.step == config.train.n_steps
        ):
            torch.cuda.empty_cache()
            logging.info(f"Save and eval checkpoint {train_state.step}...")
            if accelerator.local_process_index == 0:
                train_state.save(
                    os.path.join(config.ckpt_root, f"{train_state.step}.ckpt")
                )
            accelerator.wait_for_everyone()
            fid = eval_step(
                n_samples=30, sample_steps=50
            )  # calculate fid of the saved checkpoint
            step_fid.append((train_state.step, fid))
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    logging.info(f"Finish fitting, step={train_state.step}")
    logging.info(f"step_fid: {step_fid}")
    step_best = sorted(step_fid, key=lambda x: x[1])[0][0]
    logging.info(f"step_best: {step_best}")
    train_state.load(os.path.join(config.ckpt_root, f"{step_best}.ckpt"))
    del metrics
    accelerator.wait_for_everyone()
    eval_step(
        n_samples=config.sample.n_samples, sample_steps=config.sample.sample_steps
    )


# cfg_path = "configs/lfm_mmcelebahq256_uvit_small_deep16.py"

# cfg_path = "configs/lfm_mmcelebahq256_unet_large.py"

cfg_path = "configs/lfm_mscoco_unet_from_in256.py"

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", cfg_path, "Training configuration.", lock_config=False
)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work unit directory.")


def get_config_name():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith("--config="):
            return Path(argv[i].split("=")[-1]).stem
    return "noneconfig"


def get_hparams(config):
    argv = sys.argv
    lst = [config.nnet.name]
    for i in range(1, len(argv)):
        assert "=" in argv[i]
        if argv[i].startswith("--config.") and not argv[i].startswith(
            "--config.dataset.path"
        ):
            hparam, val = argv[i].split("=")
            hparam = hparam.split(".")[-1]
            if hparam.endswith("path"):
                val = Path(val).stem
            lst.append(f"{hparam}={val}")
    hparams = "-".join(lst)
    if hparams == "":
        hparams = "default"
    return hparams


def main(argv):
    version_str = "v2"  # v1 should be removed, I forgot to turn off the CFG flag in v1
    config = FLAGS.config

    config.config_name = get_config_name()
    config.hparams = "-".join(
        [version_str, config.nnet.name, config.dataset.name, get_hparams(config)]
    )
    config.workdir = FLAGS.workdir or os.path.join(
        "workdir", config.config_name, config.hparams
    )
    config.ckpt_root = os.path.join(config.workdir, "ckpts")
    config.sample_dir = os.path.join(config.workdir, "samples")
    train(config)


if __name__ == "__main__":
    app.run(main)
