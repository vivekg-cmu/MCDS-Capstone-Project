import os
from typing import *
import torch
import random
import numpy as np
from pytorch_lightning import Trainer
from loguru import logger
from model import Classifier
from argparse import ArgumentParser
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint


def train(config):

    logger.info(config)

    np.random.seed(42)
    random.seed(42)

    if torch.cuda.is_available():
        torch.backends.cuda.deterministic = True
        torch.backends.cuda.benchmark = False

    model = Classifier(config)
    infusion = None if "infusion" not in config else config["infusion"]
    k = None if "k" not in config else config["k"]
    train_data_portion = "partial" if "last" in config["train_x"] else "full"
    ckpt_callback = ModelCheckpoint(filepath="ckpts/infusion{}-k{}-{}/".format(infusion, k, train_data_portion), 
                                    monitor='val_acc', mode='max', verbose=True, save_top_k=1, period=0)
    trainer = Trainer(
        gradient_clip_val = 0,
        num_nodes=1,
        log_gpu_memory="all",
        log_save_interval=100,
        row_log_interval=1,
        weights_summary= 'top',
        num_sanity_val_steps=5,
        progress_bar_refresh_rate=1,
        min_epochs=1,
        max_epochs=config["max_epochs"],
        accumulate_grad_batches=config["accumulate_grad_batches"],
        val_check_interval=0.25,
        checkpoint_callback=ckpt_callback,
        gpus=None if not torch.cuda.is_available() else [i for i in range(torch.cuda.device_count())],  
        distributed_backend="ddp",
        precision=config["precision"],
        amp_level='O2',
        resume_from_checkpoint=None,
    )
    trainer.fit(model)

    pass


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--config_path', default=None)
    args = parser.parse_args()
    config_path = args.config_path
    print("config_path:", config_path)
    with open(config_path) as fp:
        config = yaml.full_load(fp)
    print("config:", config)
    
    train(config)
