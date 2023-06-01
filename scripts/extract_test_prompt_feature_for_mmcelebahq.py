import sys, os

path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)


import numpy as np
import libs.autoencoder
import libs.clip


def main():
    prompts = [
        "This smiling person has arched eyebrows, bangs, and wavy hair.",
        "The person has arched eyebrows, oval face, and high cheekbones. She wears lipstick.",
        "This attractive person has arched eyebrows, black hair, and high cheekbones.",
        "She wears lipstick. She has wavy hair, bangs, and high cheekbones. She is young, and smiling.",
        "This woman has wavy hair, and bangs. She is wearing lipstick.",
        "This person wears heavy makeup. She has wavy hair, black hair, oval face, arched eyebrows, and bangs. She is smiling, and young.",
        "She is attractive and has high cheekbones.",
        "She wears lipstick. She has bangs, arched eyebrows, wavy hair, and high cheekbones. She is young, and attractive.",
        "The person is wearing lipstick. She has wavy hair, black hair, and arched eyebrows. She is attractive, and smiling.",
        "This woman wears heavy makeup. She has arched eyebrows, and black hair.",
    ]

    device = "cuda"
    clip = libs.clip.FrozenCLIPEmbedder()
    clip.eval()
    clip.to(device)

    save_dir = f"assets/datasets/mmcelebahq256_features_withcaptioncontext/run_vis"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print("Warning: save_dir already exists. Overwriting...")

    latent = clip.encode(prompts)
    for i in range(len(latent)):
        c = latent[i].detach().cpu().numpy()
        np.save(os.path.join(save_dir, f"{i}.npy"), (prompts[i], c))


if __name__ == "__main__":
    main()
