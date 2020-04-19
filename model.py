import os
import pathlib
from typing import *
from itertools import cycle, chain

import torch
import pytorch_lightning as pl
import torch.nn as nn
import pandas as pd
import numpy as np
from loguru import logger
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup

class ClassificationDataset(Dataset):

    def __init__(self, instances):

        self.instances = instances

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]


class Classifier(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.hparams = config
        self.infusion = None if "infusion" not in self.hparams else self.hparams["infusion"]
        self.root_path = pathlib.Path(__file__).parent.absolute()
        self.embedder = AutoModel.from_pretrained(config["model"], cache_dir=self.root_path / "model_cache")
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"], cache_dir=self.root_path / "model_cache", use_fast=False)

        self.embedder.train()
        self.label_offset = 0
        self.classifier = nn.Linear(self.embedder.config.hidden_size, 1, bias=True)
        print("batch size:", self.hparams["batch_size"])

        self.loss = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")

        self.classifier.weight.data.normal_(mean=0.0, std=self.embedder.config.initializer_range)
        self.classifier.bias.data.zero_()

        # if self.infusion == "wsum":
        #     self.weight_layer = nn.Linear(self.embedder.config.hidden_size, 1, bias=True)
        #     self.weight_layer.weight.data.normal_(mean=0.0, std=self.embedder.config.initializer_range)
        #     self.weight_layer.bias.data.zero_()

    def forward(self, batch):

        assert len(batch["input_ids"].shape) == 2, "LM only take two-dimensional input"
        assert len(batch["attention_mask"].shape) == 2, "LM only take two-dimensional input"
        assert len(batch["token_type_ids"].shape) == 2, "LM only take two-dimensional input"

        # print('before: batch["token_type_ids"]:', batch["token_type_ids"])
        if "roberta" in self.hparams["model"]:
            self.embedder.embeddings.token_type_embeddings = nn.Embedding(2, self.embedder.config.hidden_size)
        # batch["token_type_ids"] = None if "roberta" in self.hparams["model"] else batch["token_type_ids"]
        print('after: batch["token_type_ids"]:', batch["token_type_ids"])
        print('after: batch["token_type_ids"].shape:', batch["token_type_ids"].shape)

        results = self.embedder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                token_type_ids=batch["token_type_ids"])
        print('batch["input_ids"]:', batch["input_ids"].shape)
        print("batch labels:", batch["labels"])

        token_embeddings, *_ = results
        # print("tokken_embeddings:", token_embeddings.shape)
        token_embeddings = token_embeddings.mean(dim=1)
        # print("tokken_embeddings:", token_embeddings.shape)
        if self.infusion == "sum":
            # seq_len = token_embeddings.shape[1]
            hidden_dim = token_embeddings.shape[1]
            # print("hidden_dim:", hidden_dim)
            token_embeddings = token_embeddings.reshape(-1, self.hparams["k"], hidden_dim).sum(dim=1)
            # print("tokken_embeddings:", token_embeddings.shape)
        # elif self.infusion == "wsum":
        #     weights = self.weight_layer(token_embeddings)

        logits = self.classifier(token_embeddings).squeeze(dim=1)
        # print('logits.shape:', logits.shape)

        if self.infusion == "max":
            logits = logits.reshape(-1, self.hparams["k"]).max(dim=1).values
            # print('logits.shape:', logits.shape)
        logits = logits.reshape(-1, batch["num_choice"])

        # print('logits.shape:', logits.shape)

        return logits

    def training_step(self, batch, batch_idx):

        logits = self.forward(batch)
        loss = self.loss(logits, batch["labels"])
        if self.trainer and self.trainer.use_dp:
            loss = loss.unsqueeze(0)
        return {
            "loss": loss
        }

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = self.loss(logits, batch["labels"])
        if self.trainer and self.trainer.use_dp:
            loss = loss.unsqueeze(0)
        return {
            'val_loss': loss,
            "val_batch_logits": logits,
            "val_batch_labels": batch["labels"],
        }

    def validation_end(self, outputs):

        val_loss_mean = torch.stack([o['val_loss'] for o in outputs]).mean()
        val_logits = torch.cat([o["val_batch_logits"] for o in outputs])
        val_labels = torch.cat([o["val_batch_labels"] for o in outputs])
        return {
            'val_loss': val_loss_mean,
            "val_acc": torch.sum(val_labels == torch.argmax(val_logits, dim=1)) / (val_labels.shape[0] * 1.0),
            "progress_bar": {
                'val_loss': val_loss_mean,
                "val_acc": torch.sum(val_labels == torch.argmax(val_logits, dim=1)) / (val_labels.shape[0] * 1.0)
            }
        }

    def configure_optimizers(self):

        t_total = len(self.train_dataloader()) // self.hparams["accumulate_grad_batches"] * self.hparams["max_epochs"]

        optimizer = AdamW(self.parameters(), lr=float(self.hparams["learning_rate"]), eps=float(self.hparams["adam_epsilon"]))

        return optimizer

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.dataloader(self.root_path / self.hparams["train_x"], self.root_path / self.hparams["train_y"]),
                          batch_size=self.hparams["batch_size"], collate_fn=self.collate)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.dataloader(self.root_path / self.hparams["val_x"], self.root_path / self.hparams["val_y"]),
                          batch_size=self.hparams["batch_size"], collate_fn=self.collate)


    def dataloader(self, x_path: Union[str, pathlib.Path], y_path: Union[str, pathlib.Path] = None):
        print("x_path:", x_path)
        df = pd.read_json(x_path, lines=True)
        if y_path:
            labels = pd.read_csv(y_path, sep='\t', header=None).values.tolist()
            self.label_offset = np.asarray(labels).min()
            df["label"] = np.asarray(labels) - self.label_offset

        k = None if "k" not in self.hparams else self.hparams["k"]
        infusion_type = None if "infusion" not in self.hparams else self.hparams["infusion"]
        df["text"] = df.apply(self.transform(self.hparams["formula"], k, infusion_type), axis=1)
        print(df.head())  # 'goal': goal, 'text': [(goal, sol1), (goal, sol2)]
        return ClassificationDataset(df[["text", "label"]].to_dict("records"))


    @staticmethod
    def transform(formula, k=None, infusion=None):

        def warpper(row):

            context, choice_names = formula.split("->")
            # (goal -> sol1|sol2)
            context = context.split("+")
            choice_names = choice_names.split("|")
            context = " ".join(row[x.strip()] for x in context)
            choices = [row[x.strip()] for x in choice_names]

            if infusion == 'concat':
                knowledges = ["\n".join(row[x.strip()+'_knowledge'][:k]) for x in choice_names]
                context_choices = [context + " " + choice for choice in choices]
                return list(zip(knowledges, context_choices))
            elif infusion == "max" or infusion == "sum":
                knowledges = [row[x.strip()+'_knowledge'][:k] for x in choice_names]
                context_choices = [context + " " + choice for choice in choices]
                k_context_choices = [zip(knowledge, cycle([cc])) for knowledge, cc in zip(knowledges, context_choices)]
                return list(chain.from_iterable(k_context_choices))
            elif not infusion:
                return list(zip(cycle([context]), choices))
            else:
                exit("Knowledge infusion method {} not supported".format(infusion))

        return warpper


    def collate(self, examples):

        batch_size = len(examples)
        num_choice = len(examples[0]["text"])
        if "infusion" in self.hparams and self.hparams["infusion"] != "concat":
            num_choice //= self.hparams["k"]
        # print([len(example["text"]) for example in examples])

        pairs = [pair for example in examples for pair in example["text"]]
        results = self.tokenizer.batch_encode_plus(pairs, add_special_tokens=True,
                                                   max_length=self.hparams["max_length"], return_tensors='pt',
                                                   return_attention_masks=True, pad_to_max_length=True)
        # print('results["input_ids"].shape:', results["input_ids"].shape)

        k = 1 if "k" not in self.hparams or self.hparams["infusion"] == "concat" else self.hparams["k"]
        assert results["input_ids"].shape[0] == batch_size * num_choice * k, \
            f"Invalid shapes {results['input_ids'].shape} {batch_size, num_choice}"

        return {
            "input_ids": results["input_ids"],
            "attention_mask": results["attention_mask"],
            "token_type_ids": results["token_type_ids"],
            "labels": torch.LongTensor([e["label"] for e in examples]) if "label" in examples[0] else None,
            "num_choice": num_choice
        }


if __name__ == "__main__":

    x_path = "data/piqa/train-knowledge-last100.jsonl"
    y_path = "data/piqa/train-labels-last100.lst"
    k = 2
    infusion = "max"
    print("x_path:", x_path)

    df = pd.read_json(x_path, lines=True)
    if y_path:
        labels = pd.read_csv(y_path, sep='\t', header=None).values.tolist()
        df["label"] = np.asarray(labels)

    df["text"] = df.apply(Classifier.transform("goal -> sol1|sol2", k, infusion), axis=1)
    print(df.head())
    print(df['text'][0])
    print(df[["text", "label"]].to_dict("records")[:2])
    dataset = ClassificationDataset(df[["text", "label"]].to_dict("records"))
    print(len(dataset.instances[0]['text']))
    ll = [len(d['text']) for d in dataset.instances]
    print(ll)
    print(sum(ll) / len(ll))
