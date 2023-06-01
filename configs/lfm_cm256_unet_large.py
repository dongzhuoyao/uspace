import ml_collections
import os

from configs.config_utils import get_epoch_id_from_path, update_config


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1230
    config.z_shape = (4, 32, 32)
    config.vis_num = 16

    config.pretrained_path = (
        "assets/latent-diffusion/cin/model.ckpt"  # imagenet256_uvit_huge.pth
    )

    config.autoencoder = d(pretrained_path="assets/stable-diffusion/autoencoder_kl.pth")

    config.train = d(
        n_steps=300000,
        batch_size=24,
        mode="uncond",
        log_interval=50,
        eval_interval=1000,
        save_interval=10000,
    )

    config.optimizer = d(
        name="adam",  # default adamw
        lr=1e-4,
        weight_decay=0.00,  # 0.03 default uvit
        betas=(0.9, 0.999),  # (0.99, 0.999),
    )

    config.lr_scheduler = d(name="customized", warmup_steps=0)

    config.nnet = d(
        name="unet_t2i",
        image_size=32,
        in_channels=4,
        out_channels=4,
        model_channels=256,
        attention_resolutions=[4, 2, 1],
        num_res_blocks=2,
        channel_mult=[1, 2, 4],
        num_head_channels=32,
        use_spatial_transformer=True,
        transformer_depth=1,
        context_dim=768,
        use_fp16=True,
        use_latent1d=0,
    )

    config.dynamic = d(
        sigma_min=1e-4,
    )

    config.dataset = d(
        name="celebamask256_features_cond",
        path="assets/datasets/celebamask256_features_with_supervision",
        cfg=False,
        p_uncond=0.15,
    )

    config.dl = d(
        num_workers=8,
        diss_num_workers=2,
    )

    config.sample = d(
        sample_steps=50,
        n_samples=16,
        mini_batch_size=8,  # the decoder is large
        scale=0.4,
        path="samples/" + config.dataset.name,
    )

    # _scales = [-500, -400, -300, -200, 100, 50, 10, 0, 10, 50, 100, 200, 300, 400, 500]
    # _scales = [-100, -50, -10, -5, 0, 5, 10, 50, 100]
    _scales = [-2, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 2]
    _scales = [1 * s for s in _scales]

    # 4, bald
    # 7, big nose
    # 20, Male
    # 15, Eyeglasses
    # 31, Smiling
    # 22, Mustache
    # 39, young
    # 8 Black_Hair
    # 9 Blond_Hair
    # 33 Wavy_Hair

    config.dissection = d(
        has_attr=True,
        dissect_task="hspace_unet",
        dissect_name=None,  # "write_pca",
        n_samples=100,
        mini_batch_size=10,
        ckpt_path_to_dissect="workdir/lfm_cm256_unet_large/v1-unet-celebamask256_features-latent0-batch_size=24/ckpts/180000.ckpt/nnet.pth",
        fixed_z_path=None,  # None or the root path for npy of the fixed z, used for real image editing purpose.
        write_path_root=None,
        vis_path=None,
        # "assets/dissections/ffhq256_features_latents.npy"
        write_scales=_scales,
        ith_component=7,  # 1, #ith component in the pca space
        ith_attr=31,
        pca_n=100,  # number of components in the pca space
        t_edit=0.5,
        edit_loc=None,
        solver_kwargs=d(
            solver="fixed",  # fixed, adaptive, fixadp
            solver_fix="euler",  # Fixed solvers (euler, midpoint, rk4)
            solver_fix_step=0.01,
            solver_adaptive="dopri5",  # dopri5, rk4
            solver_adaptive_prec=0.01,
        ),
    )

    config = update_config(config)

    return config
