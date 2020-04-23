import torch
import random
import yaml
import numpy as np
from typing import *
from pytorch_lightning import Trainer
from loguru import logger
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint

from model import Classifier


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
    if "checkpoint" in config:
        resume = config["checkpoint"].split("/")[-1].split("_")[0]
        filepath = "ckpts/infusion{}-k{}-{}-precision{}-dropout{}-resume{}/".format(
            infusion, k, train_data_portion, config["precision"], config["dropout"], resume)
    else:
        filepath = "ckpts/infusion{}-k{}-{}-precision{}-dropout{}/".format(
            infusion, k, train_data_portion, config["precision"], config["dropout"])
    ckpt_callback = ModelCheckpoint(filepath=filepath, monitor='val_acc', 
                                    mode='max', verbose=True, save_top_k=1, period=0)
    trainer = Trainer(
        gradient_clip_val=0,
        num_nodes=1,
        log_gpu_memory="all",
        log_save_interval=100,
        row_log_interval=1,
        weights_summary='top',
        num_sanity_val_steps=5,
        progress_bar_refresh_rate=1,
        min_epochs=1,
        max_epochs=config["max_epochs"],
        accumulate_grad_batches=config["accumulate_grad_batches"],
        val_check_interval=0.25,
        checkpoint_callback=ckpt_callback,
        gpus=None if not torch.cuda.is_available() else [i for i in range(torch.cuda.device_count())],
        dropout=config["dropout"],
        distributed_backend="ddp",
        precision=config["precision"],
        amp_level='O2',
        resume_from_checkpoint=None if "checkpoint" not in config else config["checkpoint"]
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
