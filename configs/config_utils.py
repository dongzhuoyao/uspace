import os
from configs.configs_utils_common import construct_solver_desc

from tools.utils_attr import get_attr_name_from_attr_id


def get_epoch_id_from_path(path):
    return int(path.split("/")[-2].split(".")[0])


def update_config(config):
    config.dissection.update(
        dataset_name=config.dataset.name,
    )

    _attr_name = get_attr_name_from_attr_id(
        config.dissection.ith_attr, config.dissection.dataset_name
    )
    config.dissection.update(
        vis_path=os.path.join(
            f"dissections_vis_v4",
            "_".join(
                [
                    config.nnet.name,
                    config.dataset.name,
                    f"{config.dissection.dissect_name}",
                    f"{_attr_name}",
                    f"attr{int(config.dissection.ith_attr)}",
                    f"latent{int(config.nnet.use_latent1d)}",
                    f"fixz{0 if config.dissection.fixed_z_path is None else 1}",
                    f"ep{get_epoch_id_from_path(config.dissection.ckpt_path_to_dissect)}",
                    f"{construct_solver_desc(**config.dissection.solver_kwargs)}",
                    f"t_edit{config.dissection.t_edit}",
                    f"{config.dissection.edit_loc}",
                    f"com{int(config.dissection.ith_component)}",
                ]
            ),
        )  # required field
    )

    config.dissection.update(
        read_path_root=f"mid_feat_with_latentz_ssdstore/"
        + "_".join(
            [
                config.nnet.name,
                "realimg",
                config.dataset.name,
                f"ep{get_epoch_id_from_path(config.dissection.ckpt_path_to_dissect)}",
                f"{construct_solver_desc(**config.dissection.solver_kwargs)}",
                f"{config.dissection.edit_loc}",
                f"n{config.dissection.n_samples}",
            ]
        )
    )

    return config
