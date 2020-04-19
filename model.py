import os
import pathlib
from typing import *
from itertools import cycle, chain
from more_itertools import split_after

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
        self.type_vocab_size = 3 if "infusion" in self.hparams else 2  # encode as "k </s> q </s> a" so will have 3 type of tokens
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

        self.embedder.embeddings.token_type_embeddings = nn.Embedding(self.type_vocab_size, self.embedder.config.hidden_size)
        # batch["token_type_ids"] = None if "roberta" in self.hparams["model"] else batch["token_type_ids"]
        print('batch["token_type_ids"].unique():', batch["token_type_ids"].unique())
        print('batch["token_type_ids"].shape:', batch["token_type_ids"].shape)

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
                return list(zip(knowledges, cycle([context]), choices))
            elif infusion == "max" or infusion == "sum":
                knowledges = [row[x.strip()+'_knowledge'][:k] for x in choice_names]
                k_context_choices = [zip(knowledge, cycle([context]), cycle([choice])) for knowledge, choice
                                     in zip(knowledges, choices)]
                return list(chain.from_iterable(k_context_choices))
            elif not infusion:
                return list(zip(cycle([context]), choices))
            else:
                exit("Knowledge infusion method {} not supported".format(infusion))

        return warpper

    @staticmethod
    def get_token_type_ids(input_ids, sep_id, max_segs):
        token_type_ids = []
        ctn, token_type = 0, 0
        for idx, input_id in enumerate(input_ids):
            ctn += 1
            if input_id == sep_id:
                token_type_ids += [token_type for _ in range(ctn)]
                ctn = 0
                token_type += 1
                max_segs -= 1
                if max_segs == 1:
                    # only one seg left, so add type ids of the rest directly
                    rest_len = len(input_ids) - (idx + 1)
                    token_type_ids += [token_type for _ in range(rest_len)]
                    return token_type_ids
        if ctn != 0:
            token_type_ids += [token_type for _ in range(ctn)]
        return token_type_ids

    def collate(self, examples):

        batch_size = len(examples)
        num_choice = len(examples[0]["text"])
        if "infusion" in self.hparams and self.hparams["infusion"] != "concat":
            num_choice //= self.hparams["k"]
        # print([len(example["text"]) for example in examples])

        if "infusion" in self.hparams:
            concated_triplets = [tokenizer.sep_token.join(triplet) for example in examples for triplet in example["text"]]
            results = self.tokenizer.batch_encode_plus(concated_triplets, add_special_tokens=True,
                                                  max_length=self.hparams["max_length"], return_tensors='pt',
                                                  return_attention_masks=True, pad_to_max_length=True)
            results["token_type_ids"] = [torch.tensor(self.get_token_type_ids(input_ids, self.tokenizer.sep_token_id, 3))
                                         for input_ids in results["input_ids"]]
        else:
            pairs = [pair for example in examples for pair in example["text"]]
            results = self.tokenizer.batch_encode_plus(pairs, add_special_tokens=True,
                                                       max_length=self.hparams["max_length"], return_tensors='pt',
                                                       return_attention_masks=True, pad_to_max_length=True)
        print('results["input_ids"].shape:', results["input_ids"].shape)
        print('results["token_type_ids"]:', results["token_type_ids"])

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
    # ll = [len(d['text']) for d in dataset.instances]
    # print(ll)
    # print(sum(ll) / len(ll))

    tokenizer = AutoTokenizer.from_pretrained("roberta-large", cache_dir="model_cache", use_fast=False)
    examples = dataset.instances[:4]
    concated_triplets = [tokenizer.sep_token.join(triplet) for example in examples for triplet in example["text"]]
    print(concated_triplets)
    # concated_triplets = ["knowledge </s> question </s> answer", "knowledge </s> question </s> answer right"]
    res = tokenizer.batch_encode_plus(concated_triplets,
                                      add_special_tokens=True,
                                      max_length=256, return_tensors='pt',
                                      return_attention_masks=True, pad_to_max_length=True
                                      )
    print(type(res["token_type_ids"]))
    print(res)
    print(tokenizer.sep_token_id)
    input_ids = res["input_ids"]

    "<s> knowledge </s> question </s> answer </s>"
    "0 xxx 2 xxx 2 xxx 2 1"
    def get_seg_lens(input_ids, sep_id, max_segs):
        ctn = 0
        for idx, input_id in enumerate(input_ids):
            ctn += 1
            if input_id == sep_id:
                yield ctn
                ctn = 0
                max_segs -= 1
                if max_segs == 1:
                    # only one seg left, so calculate the length directly
                    yield len(input_ids) - (idx + 1)
                    return
        if ctn != 0:
            yield ctn

    print(input_ids[0])
    ttids = Classifier.get_token_type_ids(input_ids[0], tokenizer.sep_token_id, 3)
    print(ttids)
    print(list(get_seg_lens(input_ids[0], tokenizer.sep_token_id, 3)))
    print(torch.tensor(ttids))


# ("knowledge question answer", "knowledge question answer 2")
#     [101, 1293, 1106, 1294, 22591, 6112, 16399, 1116, 102, 139,
#      15432, 22591, 6112, 8672, 1468, 1114, 13552, 2949, 1105, 188,
#      1643, 4854, 12767, 1114, 6870, 117, 18700, 1105, 9490, 3152,
#      7317, 1183, 119, 3299, 1114, 17053, 25138, 1105, 188, 8167,
#      23372, 1174, 22572, 23372, 1813, 119, 18757, 2391, 126, 1904,
#      1120, 3434, 4842, 143, 119, 102, 0, 0, 0, 0]
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#     [4919, 284, 787, 2603, 10872, 48813, 9414, 1530, 2603, 10872, 8469, 364, 351, 19450, 3056, 290, 45799, 351, 8268,
#      11, 13385, 290, 16577, 8278, 6874, 13, 5849, 351, 26790, 26876, 290, 37624, 269, 44937, 13, 38493, 642, 2431, 379,
#      7337, 7370, 376, 13]
#     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#      1, 1, 1, 1, 1]
