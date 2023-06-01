import sys, os

path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)

import torch
import os
import numpy as np
import libs.autoencoder
import libs.clip


clip = libs.clip.FrozenCLIPEmbedder()
clip.eval()
clip.to("cuda")


captions = [
    "I love you.",
    "I love you China.",
    "A green train is coming down the tracks.",
    "A group of skiers are preparing to ski down a mountain.",
]
latent = clip.encode(captions)
print(captions)
print(latent.shape)
