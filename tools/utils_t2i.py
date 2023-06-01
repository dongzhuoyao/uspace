import sys, os

path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)

import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel
import torch
import os
import numpy as np
import libs.autoencoder
import libs.clip
from PIL import Image
import torchvision
from absl import logging
from tools.utils_vis import _PADDING, pretty_datetime
from tools import ptp_utils
from einops import rearrange

IMG_TOKEN_NUM = 256
TIME_TOKEN_NUM = 1
CONTEXT_TOKEN_NUM = 77


def caption2context(
    caption,
):
    clip = libs.clip.FrozenCLIPEmbedder()
    clip.eval()
    clip.to("cuda")

    if isinstance(caption, str):
        _context = clip.encode([caption])
        return _context
    elif isinstance(caption, list):
        _context = clip.encode(caption)
        return _context
    else:
        raise ValueError(f"unknown caption type {type(caption)}")


def save_samplesonly_with_caption_4ablationscale(
    _data_recovered, captions, save_dir, _desc=None, **kwargs
):
    os.makedirs(save_dir, exist_ok=True)

    img_vis = _data_recovered
    img_vis = ((img_vis + 1) * 0.5).clamp(0, 1)
    os.makedirs(save_dir, exist_ok=True)
    _desc = _desc if _desc is not None else ""
    _shortname = f"{pretty_datetime()}_{_desc}_{captions[0]}"[:200]
    torchvision.utils.save_image(
        img_vis,
        os.path.join(
            save_dir,
            _shortname + ".png",
        ),
        padding=_PADDING,
        pad_value=1.0,
    )
    with open(
        os.path.join(
            save_dir,
            _shortname + ".txt",
        ),
        "w",
    ) as f:
        f.write("_".join(captions))

    print(save_dir)


def save_images_with_caption(
    _real_data, _data_recovered, captions, save_dir, multiply_desc=None, **kwargs
):
    os.makedirs(save_dir, exist_ok=True)
    for i, (_img, _img_edited, _caption) in enumerate(
        zip(_real_data, _data_recovered, captions)
    ):
        img_vis = torch.cat(
            (
                _img.unsqueeze(0),
                _img_edited.unsqueeze(0),
            ),
            dim=0,
        )
        img_vis = ((img_vis + 1) * 0.5).clamp(0, 1)
        os.makedirs(save_dir, exist_ok=True)
        multiply_desc = multiply_desc if multiply_desc is not None else ""
        _shortname = f"{pretty_datetime()}_{multiply_desc}_{_caption}"[:200]
        torchvision.utils.save_image(
            img_vis,
            os.path.join(
                save_dir,
                _shortname + ".png",
            ),
            padding=_PADDING,
            pad_value=1.0,
        )
        with open(
            os.path.join(
                save_dir,
                _shortname + ".txt",
            ),
            "w",
        ) as f:
            f.write(_caption)

    print(save_dir)


def get_phrase_ids_from_caption(_caption, _phrase):
    clip = libs.clip.FrozenCLIPEmbedder()
    clip.eval()
    clip.to("cuda")
    _res = []
    for _phr in _phrase.split(" "):
        _ids = clip.get_word_inds(text=_caption, word_place=_phr)
        _res.append(_ids)
    return np.concatenate(_res, axis=0).astype(np.int64)


def is_word_in_sentence(_word, _sentence):
    _sentence, _word = _sentence.lower(), _word.lower()

    _words = [s.strip() for s in _sentence.split(" ")]
    if _word in _words:
        return True
    else:
        return False


def is_phrase_in_sentence(_phrase, _sentence):
    _sentence, _phrase = _sentence.lower(), _phrase.lower()
    if _phrase in _sentence.lower().strip():
        return True
    else:
        return False


def vis_attention_map(attention_map, timestep_digit, origin_size=256, **kwargs):
    if timestep_digit in [
        "0.10",
        "0.20",
        "0.30",
        "0.40",
        "0.50",
        "0.60",
        "0.70",
        "0.80",
        "0.90",
    ]:
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        vis_am_path = kwargs.get("vis_am_path", None)
        _block_id = kwargs["_counter"]["block_id"]
        # [10, 8, 334, 334],attn = attn.softmax(dim=-1)
        ## token order: time_token(1), context_token(77), x(256)
        _bs, _head, _token, _ = attention_map.shape

        prompts = kwargs["caption_list"]  # TODO, "a photo of a cat"

        for select in range(_bs):
            ############################
            _am = attention_map[select].clone().cpu()
            _am = _am.mean(dim=0)  # [334, 334]
            _am_t2i = _am[
                TIME_TOKEN_NUM + CONTEXT_TOKEN_NUM :,
                TIME_TOKEN_NUM : TIME_TOKEN_NUM + CONTEXT_TOKEN_NUM,  # TO check
            ]  # [256,77]
            _am_t2i = rearrange(_am_t2i, "(a b) t -> a b t", a=16, b=16)

            tokens = tokenizer.encode(prompts[select])
            decoder = tokenizer.decode
            attention_maps = _am_t2i
            images = []
            for i in range(len(tokens)):
                image = attention_maps[:, :, i]
                image = 255 * image / image.max()
                image = image.unsqueeze(-1).expand(*image.shape, 3)
                image = image.numpy().astype(np.uint8)
                image = np.array(
                    Image.fromarray(image).resize((origin_size, origin_size))
                )
                image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
                images.append(image)
            img_pil = ptp_utils.view_images(np.stack(images, axis=0))
            os.makedirs(vis_am_path, exist_ok=True)
            img_pil.save(
                os.path.join(
                    vis_am_path,
                    f"{prompts[select]}_block{_block_id}_time{timestep_digit}.png",
                )
            )


def _p2p_rescale(
    attention_map,
    target_context_ids,
    p2p_multiplier=0,
):
    # print(_cap)
    if isinstance(p2p_multiplier, int) or isinstance(p2p_multiplier, float):
        _p2p_rescaler_list = [p2p_multiplier] * len(target_context_ids)
    elif isinstance(p2p_multiplier, list):
        _p2p_rescaler_list = p2p_multiplier
    else:
        raise ValueError(f"unknown p2p_multiplier {p2p_multiplier}")

    for _id, _target_context_id in enumerate(target_context_ids):
        if len(_target_context_id) > 0:
            _target_ids = torch.from_numpy(_target_context_id)
            assert int(torch.any(_target_ids)) < 77
            _target_ids = _target_ids + TIME_TOKEN_NUM
            if False:
                attention_map[_id, :, -IMG_TOKEN_NUM:, _target_ids] = (
                    attention_map[_id, :, -IMG_TOKEN_NUM:, _target_ids]
                    * _p2p_rescaler_list[_id]
                )  # as later, ,attn = attn.softmax(dim=-1)
            else:
                attention_map[_id, :, :, _target_ids] = (
                    attention_map[_id, :, :, _target_ids] * _p2p_rescaler_list[_id]
                )  # as later, ,attn = attn.softmax(dim=-1)

    return attention_map


def should_edit_attention_by_blockids(target_block_id, block_id):
    if isinstance(target_block_id, int):
        target_block_id = [target_block_id]
        return block_id in target_block_id
    elif isinstance(target_block_id, list):
        return block_id in target_block_id
    elif isinstance(target_block_id, str) and target_block_id == "all":
        return True
    elif target_block_id is None:
        return True  # to keep backward compatibility
    else:
        raise ValueError(f"unknown target_block_id {target_block_id}")


def real_editing_attention_map_vit(attention_map, **kwargs):
    # [10, 8, 334, 334],attn = attn.softmax(dim=-1)
    # token order: time_token(1), context_token(77), x(256)
    _bs, _head, _token, _ = attention_map.shape
    _token_kwargs = kwargs["token_kwargs"]
    if _token_kwargs["token_dissect"] == "p2p_rescale":
        if should_edit_attention_by_blockids(
            target_block_id=kwargs["block_id"], block_id=kwargs["_counter"]["block_id"]
        ):
            attention_map = _p2p_rescale(
                attention_map,
                target_context_ids=kwargs["target_context_ids"],
                p2p_multiplier=_token_kwargs["p2p_multiplier"],
            )
        return attention_map
    elif _token_kwargs["token_dissect"].startswith("lp_"):
        return attention_map

    elif _token_kwargs["token_dissect"] == "p2p_replace":
        raise NotImplementedError
    else:
        raise NotImplementedError


def editing_attention_map_vit(attention_map, timesteps, **kwargs):
    x_shape = attention_map.shape
    dissect_task = kwargs.get("dissect_task", None)
    dissect_name = kwargs.get("dissect_name", None)

    timestep_digit = f"{timesteps[0].item():.2f}"  # round(timesteps[0].item(),2)
    write_path_root = kwargs.get("write_path_root", None)
    t_edit = kwargs.get("t_edit", None)
    # print("t_edit_4debug", t_edit)

    if dissect_name in ["p2p", "local_prompt", "sampled_image_editing"]:
        _fm_direction = kwargs.get("fm_direction", None)
        if _fm_direction == "encode":
            return attention_map
        elif _fm_direction == "decode":
            if (
                "vis_am_path" in kwargs and kwargs["vis_am_path"] is not None
            ):  # visualization for p2p
                vis_attention_map(attention_map, timestep_digit, **kwargs)

            if float(timestep_digit) <= t_edit:
                attention_map = real_editing_attention_map_vit(
                    attention_map=attention_map, **kwargs
                )
            return attention_map
        else:
            raise NotImplementedError

    else:
        raise ValueError(
            f"dissect_name should be read or write, here is {dissect_name}"
        )


def local_prompt(caption_list_old, **kwargs):
    _dissect_name = kwargs["dissect_name"]
    _tk = kwargs["token_kwargs"]
    logging.warning(f"token_kwargs: {_tk}")
    if _dissect_name == "local_prompt":
        if _tk["token_dissect"] == "lp_replace":
            _c_list = [
                _c.replace(_tk["lp_replace_from"], _tk["lp_replace_to"])
                for _c in caption_list_old
            ]

        elif _tk["token_dissect"] == "lp_remove":
            _c_list = [_c.replace(_tk["lp_to_remove"], " ") for _c in caption_list_old]

        elif _tk["token_dissect"] == "lp_add":
            _c_list = [_c + " , " + _tk["lp_to_add"] for _c in caption_list_old]

        else:
            _c_list = caption_list_old

    else:
        _c_list = caption_list_old

    return _c_list
