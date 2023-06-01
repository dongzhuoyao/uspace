import torch
import torch.nn as nn
import math
import einops
import torch.utils.checkpoint
from absl import logging
import uuid, os
import numpy as np
from os.path import join as ospj


# SIZE256_BLOCK_IDS = [10, 0, 3, 5, 7, 13, 15, 17, 20]
SIZE256_BLOCK_IDS = [0]


def should_save_block_stats(block_id):
    # for resolution 256, depth=20, we save block 0, 5, 10, 15, 20
    return block_id in SIZE256_BLOCK_IDS


def should_edit(timestep_digit, t_edit):
    if timestep_digit == "0.00":
        return False

    if isinstance(t_edit, float) or isinstance(t_edit, int):
        return float(timestep_digit) <= t_edit
    elif isinstance(t_edit, str):
        if t_edit.startswith("every_"):
            _t_edit = float(t_edit.replace("every_", ""))
            return float(timestep_digit) % _t_edit == 0.0
        else:
            raise ValueError
    else:
        raise ValueError


def low_freq_logger(timesteps):
    should_log = int(timesteps[0].item() * 100) % 10 == 0
    return should_log


def should_logger_at_low_freq_blocks(timesteps, block_id):
    should_log = (
        int(timesteps[0].item() * 100) % 10 == 0 and block_id == SIZE256_BLOCK_IDS[0]
    )
    return should_log


def _read_npz_btc(npz_path, ith_ele, device):
    _delta = torch.from_numpy(np.load(npz_path)).to(device)[ith_ele]
    _delta = einops.rearrange(_delta, "t c -> 1 t c")
    return _delta


def _read_npz_bcwh(npz_path, ith_ele, device):
    def _read(npz_path, _ith_ele, device):
        _delta = torch.from_numpy(np.load(npz_path)).to(device)[_ith_ele]
        _delta = einops.rearrange(_delta, "c w h -> 1 c w h")
        return _delta

    if isinstance(ith_ele, int):
        return _read(npz_path, ith_ele, device)
    elif isinstance(ith_ele, str):
        att_list = [int(tmp) for tmp in ith_ele.split("_")]
        _delta = 0
        for _att in att_list:
            _delta += _read(npz_path, _att, device)
        return _delta / len(att_list)
    else:
        raise


def interp_ode_unet(
    ts_unit,
    timestep,
    template_str,
    ith_ele,
    device,
):
    if timestep % ts_unit == 0:
        _delta = _read_npz_bcwh(
            npz_path=template_str.replace("placeholder", f"{timestep:.2f}"),
            ith_ele=ith_ele,
            device=device,
        )
        return _delta

    else:
        ts_floor = math.floor(timestep / ts_unit) * ts_unit
        ts_ceil = math.ceil(timestep / ts_unit) * ts_unit
        if ts_floor == 0.00:  # skip 0.00, tricky engineering
            _delta = _read_npz_bcwh(
                npz_path=template_str.replace("placeholder", f"{timestep:.2f}"),
                ith_ele=ith_ele,
                device=device,
            )
            return _delta
        else:
            _delta_floor, _delta_ceil = _read_npz_bcwh(
                npz_path=template_str.replace("placeholder", f"{ts_floor:.2f}"),
                ith_ele=ith_ele,
                device=device,
            ), _read_npz_bcwh(
                npz_path=template_str.replace("placeholder", f"{ts_ceil:.2f}"),
                ith_ele=ith_ele,
                device=device,
            )
            _delta = (
                _delta_floor
                + (_delta_ceil - _delta_floor) * (timestep - ts_floor) / ts_unit
            )
            return _delta


def dissect_helper_uvit(x, timesteps, **kwargs):
    x_shape = x.shape
    dissect_task = kwargs.get("dissect_task", None)
    dissect_name = kwargs.get("dissect_name", None)

    timestep_digit = f"{timesteps[0].item():.2f}"  # round(timesteps[0].item(),2)

    t_edit = kwargs.get("t_edit", None)
    write_scale = kwargs.get("write_scale", None)

    if dissect_task == "uspace_uvit":
        if dissect_name == "read":
            read_path_root = kwargs.get("read_path_root", None)
            os.makedirs(read_path_root, exist_ok=True)

            np.save(
                ospj(
                    read_path_root,
                    f"{kwargs['batch_id']}_{timestep_digit}",
                ),
                x.detach().cpu().numpy(),  # [b,1024,8,8]
            )

        elif dissect_name == "write_attr":
            write_path_root = kwargs.get("write_path_root", None)
            ith_attr = kwargs.get("ith_attr", None)
            edit_loc = kwargs.get("edit_loc", None)
            if should_edit(timestep_digit, t_edit):
                _delta = _read_npz_bcwh(
                    npz_path=ospj(
                        write_path_root,
                        f"delta_{timestep_digit}.npy",
                    ),
                    ith_ele=ith_attr,
                    device=x.device,
                )

                if low_freq_logger(timesteps):
                    logging.warning(kwargs["write_path_root"])
                    logging.info(
                        f" x = x + _delta * {write_scale}, mean: {x.mean().item()}, std: {x.std().item()}, delta: {_delta.mean().item()}, delta_std: {_delta.std().item()}"
                    )
                x = x + _delta * write_scale
            else:
                pass  # do nothing, keep the original x
        elif dissect_name == "write_pca":
            write_path_root = kwargs.get("write_path_root", None)
            ith_component = kwargs.get("ith_component", None)
            pca_n = kwargs.get("pca_n", None)
            if should_edit(timestep_digit, t_edit):
                _delta = _read_npz_bcwh(
                    npz_path=ospj(
                        write_path_root,
                        f"pca{pca_n}_{timestep_digit}.npy",
                    ),
                    ith_ele=ith_component,
                    device=x.device,
                )

                if low_freq_logger(timesteps):
                    logging.info(
                        f" x = x + _delta * {write_scale}, mean: {x.mean().item()}, std: {x.std().item()}, delta: {_delta.mean().item()}, delta_std: {_delta.std().item()}"
                    )
                x = x + _delta * write_scale
            else:
                pass  # do nothing, keep the original x
        else:
            raise ValueError(
                f"dissect_name should be read or write, here is {dissect_name}"
            )

    return x


def dissect_helper_unet(x, timesteps, **kwargs):
    x_shape = x.shape
    dissect_task = kwargs.get("dissect_task", None)
    timestep_digit = f"{timesteps[0].item():.2f}"  # round(timesteps[0].item(),2)

    dissect_name = kwargs.get("dissect_name", None)
    write_scale = kwargs.get("write_scale", None)
    t_edit = kwargs.get("t_edit", None)

    if dissect_task == "hspace_unet":
        if int(timesteps[0].item() * 100) % 10 == 0:
            logging.info(
                f"dissect_name is not None, dissecting, {dissect_task}, timesteps:{timestep_digit}"
            )

        if dissect_name == "read":
            read_path_root = kwargs.get("read_path_root", None)
            os.makedirs(write_path_root, exist_ok=True)

            if True:
                target_path = ospj(
                    read_path_root,
                    f"{kwargs['batch_id']}_{timestep_digit}",
                )
                if os.path.exists(target_path):
                    logging.info(f"skip {target_path}")
                    raise FileExistsError(f"skip {target_path}")

                np.save(
                    target_path,
                    x.detach().cpu().numpy(),  # [b,1024,8,8]
                )

        elif dissect_name == "write_pca":
            write_path_root = kwargs.get("write_path_root", None)

            ith_component = kwargs.get("ith_component", None)
            pca_n = kwargs.get("pca_n", None)

            if should_edit(timestep_digit, t_edit):  # tricky to skip '0.00'
                _npy_filename = f"pca{pca_n}_{timestep_digit}.npy"

                _delta = _read_npz_bcwh(
                    npz_path=ospj(
                        write_path_root,
                        _npy_filename,
                    ),
                    ith_ele=ith_component,
                    device=x.device,
                )

                if False or low_freq_logger(timesteps):
                    logging.info(
                        f"x = x + _delta * {write_scale}, mean: {x.mean().item()}, std: {x.std().item()}, delta: {_delta.mean().item()}, delta_std: {_delta.std().item()}, {_npy_filename}"
                    )
                x = x + _delta * write_scale
            else:
                pass  # do nothing, keep the original x
        elif dissect_name == "write_attr":
            write_path_root = kwargs.get("write_path_root", None)
            ith_attr = kwargs.get("ith_attr", None)
            if should_edit(timestep_digit, t_edit):
                if kwargs["solver_kwargs"]["solver"] != "adaptive":
                    _delta = _read_npz_bcwh(
                        npz_path=ospj(
                            write_path_root,
                            f"delta_{timestep_digit}.npy",
                        ),
                        ith_ele=ith_attr,
                        device=x.device,
                    )
                else:
                    _delta = interp_ode_unet(
                        ts_unit=kwargs["solver_kwargs"]["solver_adaptive_prec"],
                        timestep=timesteps[0].item(),
                        template_str=ospj(
                            write_path_root,
                            f"delta_placeholder.npy",
                        ),
                        ith_ele=ith_attr,
                        device=x.device,
                    )
                if False or low_freq_logger(timesteps):
                    logging.info(
                        f"x = x + _delta * {write_scale}, mean: {x.mean().item()}, std: {x.std().item()}, delta: {_delta.mean().item()}, delta_std: {_delta.std().item()}"
                    )
                x = x + _delta * write_scale
            else:
                pass  # do nothing, keep the original x
        elif dissect_name == "write_x0":
            pass
        else:
            raise ValueError("dissect_name should be read or write")
    return x
