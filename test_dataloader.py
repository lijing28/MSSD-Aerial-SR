# from pasd.dataloader.localdatasets import LocalImageDataset
from pasd.dataloader.localdatasets_drsr import LocalImageDataset
from transformers import AutoTokenizer, PretrainedConfig

resolution = 64

tokenizer = AutoTokenizer.from_pretrained(
            'checkpoints/stable-diffusion-v1-5',
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )

train_dataset = LocalImageDataset(pngtxt_dir="/data0/llj/dataset/AID/train", image_size=resolution, tokenizer=tokenizer, accelerator=None, control_type='realisr', null_text_ratio=0.5, resize_bak=True)

train_dataset.__getitem__(0)
