import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel
import numpy as np


def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [
            tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)
        ][1:-1]
        print("words_encode_4_debug", words_encode)
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(
        self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77
    ):
        super().__init__()     
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def get_word_inds(self, text: str, word_place: int):
        _inds = get_word_inds(
            text=text, word_place=word_place, tokenizer=self.tokenizer
        )
        return _inds

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        if False:
            for _text in text:
                _inds = get_word_inds(
                    text=_text, word_place="are", tokenizer=self.tokenizer
                )
                _tokenized = self.tokenizer.tokenize(_text)
                print(_text)
                print(_inds)

                print(_tokenized)
                print("*" * 10)

        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)
