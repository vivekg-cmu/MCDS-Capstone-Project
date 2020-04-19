import os
from typing import *
import hydra
import torch
import random
import numpy as np
from pytorch_lightning import Trainer
from loguru import logger
from model import Classifier
from pytorch_lightning.callbacks import ModelCheckpoint


@hydra.main(config_path="config-local.yaml")
# @hydra.main(config_path="config.yaml")
def train(config):

    logger.info(config)

    np.random.seed(42)
    random.seed(42)

    if torch.cuda.is_available():
        torch.backends.cuda.deterministic = True
        torch.backends.cuda.benchmark = False

#     ckpt_path = os.path.join(os.getcwd(), 'lightning_logs', 'version_0', 'checkpoints')
    ckpt_callback = ModelCheckpoint(filepath=os.getcwd(), monitor='val_acc', verbose=True, save_top_k=1)
    model = Classifier(config)
    trainer = Trainer(
        gradient_clip_val = 0,
        num_nodes=1,
        gpus = None if not torch.cuda.is_available() else [i for i in range(torch.cuda.device_count())],
#         gpus = [0,1],
        log_gpu_memory=True,
        show_progress_bar=True,
        accumulate_grad_batches=config["accumulate_grad_batches"],
        max_epochs=config["max_epochs"],
        min_epochs=1,
        val_check_interval=0.2,
        log_save_interval=100,
        row_log_interval=1,
        distributed_backend = "dp",
        use_amp=config["use_amp"],
        weights_summary= 'top',
        amp_level='O2',
        num_sanity_val_steps=5,
        resume_from_checkpoint=None if "checkpoint" not in config else config["checkpoint"],
        checkpoint_callback = ckpt_callback
    )
    trainer.fit(model)

    pass

if __name__ == "__main__":
    train()
