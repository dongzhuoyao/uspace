from configs.config_utils_t2i import update_config_t2i
from tools.utils_t2i import (
    caption2context,
    get_phrase_ids_from_caption,
    is_word_in_sentence,
    is_phrase_in_sentence,
    save_images_with_caption,
    save_samplesonly_with_caption_4ablationscale,
    local_prompt,
)
from flow_matching_t2i import CNF
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


from tools.utils_vis import extract_latents


def evaluate(config):
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

    dataset = get_dataset(**config.dataset_withcaptioncontext)
    train_dataset = dataset
    # train_dataset = dataset.get_split(split="test", labeled=_exp_kwargs["has_attr"])
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
        assert len(xs) == len(_batch)
        return xs

    score_model = CNF(net=nnet)

    logging.info(config.sample)

    logging.info(
        f"sample: n_samples={config.sample.n_samples}, mode={config.train.mode}, mixed_precision={config.mixed_precision}"
    )

    _iter = iter(train_dataset_loader)

    def sample_fn(input_z, **kwargs):
        _feat = score_model.decode(
            input_z,
            **kwargs,
        )
        return decode_large_batch(_feat)

    def encode_fn(vis_reversible=False, is_reverse_back=True, **_kwargs):
        _real_data = next(_iter)

        kwargs = dict(**_kwargs)

        _real_data, _caption_list = _real_data
        _context = caption2context(_caption_list)
        kwargs.update(dict(context=_context, caption_list=_caption_list))

        _real_data = _real_data.to("cuda")
        print("real_data", _real_data.shape, _real_data.dtype)
        _feat = (
            autoencoder.sample(_real_data)
            if "feature" in config.dataset_withcaptioncontext.name
            else encode(_real_data)
        )

        _z_latent = score_model.encode(
            _feat,
            **kwargs,
        )

        if is_reverse_back:
            if kwargs["dissect_name"] == "local_prompt":
                kwargs["caption_list_edited"] = local_prompt(
                    caption_list_old=_caption_list, **kwargs
                )
                kwargs["context"] = caption2context(kwargs["caption_list_edited"])
                _caption_list_4vis = [
                    f"{_old}_{_new}"
                    for _old, _new in zip(_caption_list, kwargs["caption_list_edited"])
                ]
            elif kwargs["dissect_name"] == "p2p":
                target_context_ids = []
                p2p_to_multiply = kwargs["token_kwargs"]["p2p_to_multiply"]
                p2p_multiplier = kwargs["token_kwargs"]["p2p_multiplier"]

                for _id, _cap in enumerate(_caption_list):
                    if p2p_to_multiply is not None:
                        if not is_phrase_in_sentence(p2p_to_multiply, _cap):
                            target_context_ids.append(np.array([]))
                        else:
                            target_context_ids.append(
                                get_phrase_ids_from_caption(_cap, p2p_to_multiply)
                            )  # [0, 1, 2]
                    else:
                        logging.warning(
                            f"p2p_to_multiply is None, negate index [0,1,2,3], with p2p_multiplier={p2p_multiplier}",
                        )
                        target_context_ids.append(np.array([0, 1, 2, 3]))
                kwargs["target_context_ids"] = target_context_ids

                _caption_list_4vis = _caption_list
            else:
                raise NotImplementedError
            # _caption_list_4vis = _caption_list

            _feat_recovered = score_model.decode(
                _z_latent,
                **kwargs,
            )
            _data_recovered = decode_large_batch(_feat_recovered)

            if "vis_am_path" in kwargs and kwargs["vis_am_path"] is not None:
                _save_dir = kwargs["vis_am_path"]  # put them in the same folder
                print("overwrite vis_path", _save_dir)
            else:
                _save_dir = kwargs["vis_path"]

            save_images_with_caption(
                _real_data=_real_data,
                _data_recovered=_data_recovered,
                captions=_caption_list_4vis,
                save_dir=_save_dir,
                **kwargs,
            )

        if vis_reversible:  # For debugging
            _z_feat_recovered = score_model.decode(
                _z_latent,
                **kwargs,
            )
            logging.info(
                f"save image img-size,{_feat.shape}, {_z_feat_recovered.shape}, {torch.norm(_feat).item()}, {torch.norm(_feat - _z_feat_recovered).item()}"
            )

            if "feature" not in config.dataset_withcaptioncontext.name:
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
                    "reversible_vis_t2i.png",
                    padding=0,
                    pad_value=1.0,
                )

        return _z_latent

    if _exp_kwargs.dissect_name in ["p2p", "local_prompt"]:  # for real image editing
        if _exp_kwargs["dissect_name"] == "p2p":
            target_context_ids = []
            p2p_to_multiply = _exp_kwargs["token_kwargs"]["p2p_to_multiply"]
            p2p_multiplier = _exp_kwargs["token_kwargs"]["p2p_multiplier"]
            _exp_kwargs.vis_path = _exp_kwargs.vis_path + f"_{p2p_multiplier}"

        with tempfile.TemporaryDirectory() as temp_path:
            path = config.sample.path or temp_path
            if accelerator.is_main_process:
                os.makedirs(path, exist_ok=True)
            logging.info(f"Samples are saved in {path}")

            _npz = extract_latents(
                accelerator,
                encode_fn,
                **_exp_kwargs,
            )
            logging.info(f"save latents to {_exp_kwargs.read_path_root}")
            np.save(os.path.join(_exp_kwargs.read_path_root, "latents.npy"), _npz)
    elif _exp_kwargs.dissect_name in [
        "sampled_image_editing"
    ]:  # for sampled image editing
        if config.dataset.name.startswith("mmcelebahq"):
            _caption_list = [
                "The man has brown hair",
                "The man has brown hair, he has a mustache",
                "The man has brown hair, he has a mustache, he is smiling",
                "The man has brown hair, he has a mustache, he is smiling, he has bushy eyebrows",
                "The man has brown hair, he has a mustache, he is smiling, he has bushy eyebrows, he has pale skin",
            ]
            _caption_list = [
                "He is smiling",
                "He is smiling, He has a mustache",
                "He is smiling, He has a mustache, the man has brown hair",
                "He is smiling, He has a mustache, the man has brown hair, he has bushy eyebrows",
                "He is smiling, He has a mustache, the man has brown hair, he has bushy eyebrows, he has pale skin",
            ]
            _caption_list = [
                "He has pale skin",
                "He has pale skin, he has bushy eyebrows",
                "He has pale skin, he has bushy eyebrows, he is smiling",
                "He has pale skin, he has bushy eyebrows, he is smiling, he has a mustache",
                "He has pale skin, he has bushy eyebrows, he is smiling, he has a mustache, The man has brown hair",
            ]
            _caption_list = [
                "She has brown hair.",
                "She has brown hair, she is young.",
                "She has brown hair, she is young,  she has mouth slightly open.",
                "She has brown hair, she is young,  she has mouth slightly open, she has a pointy nose.",
                "She has brown hair, she is young,  she has mouth slightly open, she has a pointy nose, she has rosy cheeks.",
            ]
        else:
            _caption_list = [
                "a man is playing basketball.",
                "a man is playing basketball, the weather is sunny.",
            ]
        single_latent = True
        sie_type = "sop_direct"
        assert sie_type in ["sop_rescale", "sop_direct", "sop_lp"]

        if sie_type in ["sop_rescale"]:
            t_edit = config.dissection.t_edit
            p2p_to_multiply = "hair"

            p2p_multiplier = [-5, -3, -1, 0, 1, 3, 5]
            p2p_multiplier = [i * 2 for i in p2p_multiplier]
            if isinstance(p2p_multiplier, list):
                assert len(_caption_list) == 1
                _caption_list = [_caption_list[0]] * len(p2p_multiplier)

            _desc = f"{p2p_to_multiply}_x_{p2p_multiplier}_tedit{t_edit}"

            _exp_kwargs["token_kwargs"].update(
                token_dissect="p2p_rescale",
                p2p_to_multiply=p2p_to_multiply,
                p2p_multiplier=p2p_multiplier,
            )
        else:
            _desc = None
        #################################
        assert len(_caption_list) > 0
        _save_dir = os.path.join(
            f"sampled_image_editing/seed{config.seed}_single_latent{int(single_latent)}"
        )

        z_shape = (config.z_shape[0], config.z_shape[1], config.z_shape[2])
        if single_latent:
            input_z = torch.randn(1, *z_shape, device=device)
            input_z = input_z.repeat(len(_caption_list), 1, 1, 1)
        else:
            input_z = torch.randn(len(_caption_list), *z_shape, device=device)
        if sie_type in ["sop_lp"]:  # used for local_prompt
            _exp_kwargs["caption_list_edited"] = local_prompt(
                caption_list_old=_caption_list, **_exp_kwargs
            )
            _exp_kwargs["context"] = caption2context(_exp_kwargs["caption_list_edited"])
            _caption_list_4vis = [
                f"{_old}_{_new}"
                for _old, _new in zip(_caption_list, _exp_kwargs["caption_list_edited"])
            ]
        elif sie_type in ["sop_direct"]:
            _exp_kwargs["context"] = caption2context(_caption_list)
            _caption_list_4vis = _caption_list
        elif sie_type in ["sop_rescale"]:  # used for p2p
            target_context_ids = []
            for _id, _cap in enumerate(_caption_list):
                if not is_word_in_sentence(p2p_to_multiply, _cap):
                    target_context_ids.append(np.array([]))
                else:
                    target_context_ids.append(
                        get_phrase_ids_from_caption(_cap, p2p_to_multiply)
                    )  # [0, 1, 2]
            _exp_kwargs["target_context_ids"] = target_context_ids

            _exp_kwargs["context"] = caption2context(_caption_list)
            _caption_list_4vis = _caption_list
        else:
            raise NotImplementedError

        _feat_recovered = score_model.decode(
            input_z,
            **_exp_kwargs,
        )
        _data_recovered = decode_large_batch(_feat_recovered)
        save_samplesonly_with_caption_4ablationscale(
            _data_recovered=_data_recovered,
            captions=_caption_list_4vis,
            save_dir=_save_dir,
            _desc=_desc,
            **_exp_kwargs,
        )
    else:
        raise NotImplementedError


########
if False:
    cfg_path = "configs/lfm_mscoco_uvit_from_in256.py"
    if False:
        dissect_name = "p2p"
        token_dissect = "p2p_rescale"
    else:
        dissect_name = "sampled_image_editing"  # "local_prompt"
        token_dissect = "lp_add"

    block_id = "all"
    edit_loc = "mid"

    write_path_root = "mid_feat_with_latentz_ssdstore/t2i_debug"

if True:
    cfg_path = "configs/lfm_mscoco_unet_from_in256.py"
    if False:
        dissect_name = "p2p"
        token_dissect = "p2p_rescale"
    else:
        dissect_name = "sampled_image_editing"  # "local_prompt"
        token_dissect = "lp_add"

    block_id = "all"
    edit_loc = "none"

    write_path_root = "mid_feat_with_latentz_ssdstore/t2i_debug"

elif False:  # ready for attention map vis
    cfg_path = "configs/lfm_mmcelebahq256_uvit_small_deep16_scratch.py"
    if False:
        dissect_name = "p2p"
        token_dissect = "p2p_rescale"
    elif True:
        dissect_name = "local_prompt"
        token_dissect = "lp_replace"
    elif False:
        dissect_name = "sampled_image_editing"  # "local_prompt"
        token_dissect = None

    block_id = "all"
    edit_loc = "mid"

    write_path_root = (
        "mid_feat_with_latentz_ssdstore/t2i_mmcelebahq256_fromcoco_scratch"
    )

elif True:
    cfg_path = "configs/lfm_mmcelebahq256_unet_large.py"
    if False:
        dissect_name = "p2p"
        token_dissect = "p2p_rescale"
    elif False:
        dissect_name = "local_prompt"
        token_dissect = "lp_replace"
    elif False:
        dissect_name = "local_prompt"
        token_dissect = "lp_add"
    elif True:
        dissect_name = "sampled_image_editing"  # "local_prompt"
        token_dissect = None

    block_id = "all"
    edit_loc = "none"

    write_path_root = (
        "mid_feat_with_latentz_ssdstore/t2i_mmcelebahq256_fromcoco_scratch"
    )

else:
    raise NotImplementedError


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
    config.nnet.use_checkpoint = False
    logging.warning(f"config.nnet.use_checkpoint {config.nnet.use_checkpoint}")
    config.nnet_path = config.dissection.ckpt_path_to_dissect
    config.dissection.seed = config.seed
    config.dissection.write_path_root = write_path_root

    logging.warning(f"seed {config.seed}")

    config.sample.n_samples = config.dissection.n_samples
    config.sample.mini_batch_size = config.dissection.mini_batch_size
    config.dissection.dissect_name = dissect_name
    config.dissection.edit_loc = edit_loc
    config.dissection.block_id = block_id
    config.dissection.token_kwargs.token_dissect = token_dissect

    config._exp_kwargs = config.dissection

    config = update_config_t2i(config)  # final step
    ###################

    evaluate(config)


if __name__ == "__main__":
    app.run(main)
