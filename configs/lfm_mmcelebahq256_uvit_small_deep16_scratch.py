import ml_collections
import os

from configs.config_utils import get_epoch_id_from_path
from configs.config_utils_t2i import update_config_t2i


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1230
    config.z_shape = (4, 32, 32)
    config.vis_num = 16

    config.pretrained_path = None  # imagenet256_uvit_huge.pth
    # imagenet512_uvit_large.pth

    config.autoencoder = d(pretrained_path="assets/stable-diffusion/autoencoder_kl.pth")

    config.train = d(
        n_steps=1000000,
        batch_size=256,
        mode="uncond",
        log_interval=10,
        eval_interval=1000,
        save_interval=10000,
    )

    config.optimizer = d(
        name="adam",  # default adamw
        lr=1e-4,
        weight_decay=0.01,  # 0.03 default uvit
        betas=(0.9, 0.999),  # (0.99, 0.999),
    )

    config.lr_scheduler = d(name="customized", warmup_steps=0)

    config.nnet = d(
        name="uvit_t2i",
        img_size=32,
        in_chans=4,
        patch_size=2,
        embed_dim=512,
        depth=16,  # 12->16
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        clip_dim=768,
        num_clip_token=77,
        use_checkpoint=True,
        use_latent1d=False,  # dump arg
    )

    config.dynamic = d(
        sigma_min=1e-4,
    )

    config.dataset = d(
        name="mmcelebahq256_features_withcaptioncontext",
        path="assets/datasets/mmcelebahq256_features_withcaptioncontext",
    )

    config.dataset_withcaptioncontext = d(
        name="mmcelebahq256_withcaptioncontext",
        root="~/data_hhd/mm-celeba-hq",
        size=256,
        is_for_dissect=True,
    )

    config.dl = d(
        num_workers=8,
        diss_num_workers=2,
    )

    config.sample = d(
        sample_steps=50,
        n_samples=16,
        mini_batch_size=8,  # the decoder is large
        path="samples/" + config.dataset.name,
    )

    # _scales = [-500, -400, -300, -200, 100, 50, 10, 0, 10, 50, 100, 200, 300, 400, 500]
    # _scales = [-100, -50, -10, -5, 0, 5, 10, 50, 100]
    _scales = [-2, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 2]
    _scales = [1 * s for s in _scales]

    config.dissection = d(
        has_attr=True,
        dissect_task="uspace_uvit",
        dissect_name=None,  # "write_pca",
        n_samples=1000,
        mini_batch_size=10,
        ckpt_path_to_dissect="workdir/lfm_mmcelebahq256_uvit_small_deep16_scratch/v2-uvit_t2i-mmcelebahq256_features_withcaptioncontext-uvit_t2i-batch_size=256-weight_decay=0-lr=2e-4/ckpts/250000.ckpt/nnet.pth",
        fixed_z_path=None,  # None or the root path for npy of the fixed z, used for real image editing purpose.
        write_path_root=None,
        vis_path=None,
        # "assets/dissections/ffhq256_features_latents.npy"
        write_scales=_scales,
        ith_component=7,  # 1, #ith component in the pca space
        pca_n=100,  # number of components in the pca space
        ith_attr=7,
        t_edit=0.3,
        edit_loc=None,
        vis_am_path="vis_am_v2",  # if not None, will visualize the attention map
        solver_kwargs=d(
            solver="fixed",  # fixed, adaptive, fixadp
            solver_fix="euler",  # Fixed solvers (euler, midpoint, rk4)
            solver_fix_step=0.01,
            solver_adaptive="dopri5",  # dopri5, rk4
            solver_adaptive_prec=0.01,
        ),
        token_kwargs=d(
            # if dissect_name="local_prompt", token_dissect="lp_add, lp_remove, lp_replace"
            # if dissect_name="p2p", token_dissect=p2p_rescale
            token_dissect=None,
            # lp_replace_from="long",
            # lp_replace_to="short",
            #
            # lp_replace_from="man",
            # lp_replace_to="woman",
            #
            lp_replace_from="heavy makeup",
            lp_replace_to="light makeup",
            lp_to_add="driving a car",
            # lp_to_add="the person is smiling",
            # lp_to_add="The person has mouth slightly open",
            lp_to_remove="smiling",
            # lp_to_remove="young",
            p2p_to_multiply="wavy hair",
            p2p_multiplier=-11,
        ),
    )

    config = update_config_t2i(config)

    return config
