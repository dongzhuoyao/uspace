import os

from configs.configs_utils_common import construct_solver_desc


def get_epoch_id_from_path(path):
    return int(path.split("/")[-2].split(".")[0])


def construct_token_desc(dissect_name, **token_kwargs):
    if dissect_name == "local_prompt":
        if token_kwargs["token_dissect"] == "lp_add":
            return f"add_{token_kwargs['lp_to_add']}"
        elif token_kwargs["token_dissect"] == "lp_remove":
            return f"remove_{token_kwargs['lp_to_remove']}"
        elif token_kwargs["token_dissect"] == "lp_replace":
            return f"replace_{token_kwargs['lp_replace_from']}_{token_kwargs['lp_replace_to']}"
        elif token_kwargs["token_dissect"] == "p2p_rescale":
            raise NotImplementedError
            return f"multi_{token_kwargs['token_to_multiply']}_by_{token_kwargs['token_multiplier']}"
        else:
            return "no-opr"
    elif dissect_name == "p2p":
        if token_kwargs["token_dissect"] == "p2p_add":
            return f"add_{token_kwargs['lp_to_add']}"
        elif token_kwargs["token_dissect"] == "p2p_remove":
            return f"remove_{token_kwargs['lp_to_remove']}"
        elif token_kwargs["token_dissect"] == "p2p_replace":
            return f"replace_{token_kwargs['lp_replace_from']}_{token_kwargs['lp_replace_to']}"
        elif token_kwargs["token_dissect"] == "p2p_rescale":
            return f"multiply_{token_kwargs['p2p_to_multiply']}_by_{token_kwargs['p2p_multiplier']}"

        else:
            return "no-opr"
    elif dissect_name == "sampled_image_editing":
        return dissect_name
    elif dissect_name is None:  # used for training
        return "nonedissect"
    else:
        raise NotImplementedError(f"unknown dissect_name {dissect_name}")


def update_config_t2i(config):
    config.dissection.update(
        dataset_name=config.dataset.name,
    )

    config.dissection.write_path_root = (
        config.dissection.write_path_root or f"write_path_root_tmp"
    )
    config.dissection.fixed_z_path = config.dissection.write_path_root + "/latents.npy"

    config.dissection.update(
        vis_path=os.path.join(
            f"dissections_vis_t2i_v3",
            "_".join(
                [
                    # config.nnet.name,
                    f"{construct_token_desc(dissect_name=config.dissection.dissect_name , **config.dissection.token_kwargs)}",
                    f"fixz{0 if config.dissection.fixed_z_path is None else 1}",
                    f"ep{get_epoch_id_from_path(config.dissection.ckpt_path_to_dissect)}",
                    f"{construct_solver_desc(**config.dissection.solver_kwargs)}",
                    config.dataset.name,
                    f"t_edit{config.dissection.t_edit}",
                ]
            ),
        )  # required field
    )

    os.makedirs(config.dissection.vis_path, exist_ok=True)

    config.dissection.update(
        read_path_root=f"mid_feat_with_latentz_ssdstore/"
        + "_".join(
            [
                config.nnet.name,
                "realimg",
                config.dataset.name,
                f"ep{get_epoch_id_from_path(config.dissection.ckpt_path_to_dissect)}",
                # f"latent{config.nnet.use_latent1d}",
                f"{config.dissection.edit_loc}",
                f"n{config.dissection.n_samples}",
                f"{construct_solver_desc(**config.dissection.solver_kwargs)}",
            ]
        )
    )

    return config
