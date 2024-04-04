
# Latent Space Editing in Transformer-based Flow Matching

<span class="author-block">
                <a href="http://taohu.me" target="_blank">Vincent Tao HU</a>,</span>
                <span class="author-block">
                  <a href="https://davzha.netlify.app/" target="_blank">David W Zhang</a>,</span>
                  <span class="author-block">
                    <a href="https://staff.fnwi.uva.nl/p.s.m.mettes/" target="_blank">Pascal Mettes</a>,
                  </span>
                  <span class="author-block">
                    <a href="http://mengtang.org/" target="_blank">Meng Tang</a>,
                  </span>
                  <span class="author-block">
                    <a href="https://scholar.google.com/citations?user=7LhjCn0AAAAJ&hl=en" target="_blank">Deli Zhao</a>,
                  </span>
                  <span class="author-block">
                    <a href="https://www.ceessnoek.info/" target="_blank">Cees G.M. Snoek</a>
                  </span>

 **AAAI 2024**

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://taohu.me/lfm/)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2312.10825)

```
@inproceedings{hulfm,
        title = {Latent Space Editing in Transformer-based Flow Matching},
        author = {Hu, Tao and Zhang, David W and Mettes, Pascal and Tang, Meng and Zhao, Deli and Snoek, Cees G.M.},
        year = {2024},
        booktitle = {AAAI},
      }
```       



### Local-Prompt 

Adding(Sequentially), Replacing, Rescaling

```bash
python dissect_lfm_t2i.py 
```



### Semantic Direction Manipulation

step1: semantic direction collection

```bash
python dissect_lfm.py 
```


step2:generate the semantic direction

```bash
python tools/utils_attr.py
```

step3:semantic steering

```bash
python dissect_lfm.py 
```



# Pretraining on datasets


### CelebAMask256

```  
CUDA_VISIBLE_DEVICES=4,5,6,7  accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train_lfm.py --config=configs/lfm_cm256_uvit_large.py --config.train.batch_size=512
```


### MM-Celeba_HQ



```
CUDA_VISIBLE_DEVICES=4,5,6,7  accelerate launch --multi_gpu --num_processes 4 --main_process_port 8839  --mixed_precision fp16 train_lfm_t2i.py --config=configs/lfm_mmcelebahq256_uvit_large.py --config.train.batch_size=512
```


### COCO

```
python scripts/extract_mscoco_feature.py #train
python scripts/extract_mscoco_feature.py #val
python scripts/extract_empty_feature.py
python scripts/extract_test_prompt_feature.py
```



```
CUDA_VISIBLE_DEVICES=4,5,6,7  accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train_lfm_t2i.py --config=configs/lfm_mscoco_uvit_from_in256.py --config.train.batch_size=256 
```


### Prepare ./assets

following [U-ViT](https://github.com/baofff/U-ViT)

- fid_stats, a dummy file
- pretrained_weights, for initialization and fine-tunig
-  stable-diffusion, need the Encoder-Decoder weight


# Environment Preparation


```bash
conda create -n lfmuvit  python=3.10
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pytorch-lightning torchdiffeq  matplotlib h5py timm diffusers accelerate loguru blobfile ml_collections
pip install hydra-core wandb einops scikit-learn --upgrade
pip install einops sklearn
pip install transformers==4.23.1 pycocotools # for text-to-image task
```




# Acknowledgement

This codebase is developed based on U-ViT, if you find this repo useful, please consider citing the following paper:

```
@inproceedings{bao2022all,
  title={All are Worth Words: A ViT Backbone for Diffusion Models},
  author={Bao, Fan and Nie, Shen and Xue, Kaiwen and Cao, Yue and Li, Chongxuan and Su, Hang and Zhu, Jun},
  booktitle = {CVPR},
  year={2023}
}
```
