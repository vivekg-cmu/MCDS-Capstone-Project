# Run Baselines with Bert-like models

## Dependencies

Install apex if you want to use half precision: https://github.com/NVIDIA/apex


```bash
pip install -r requirements.txt
```

## Download datasets

Download desired data sets into `data`. For example, the piqa data set is under `data/piqa` with the following structure
```
data
└── piqa
    ├── tests.jsonl
    ├── train.jsonl
    ├── train-labels.lst
    ├── valid.jsonl
    └── valid-labels.lst
```

## Train

Modify `config.yaml` as you like and run `python train.py` to train a model. It loads the config file and outputs all the logs/checkpoints in `outputs`

## Eval

### Get predictions without evaluation
```bash
python eval.py \
    --input_x data/piqa/valid.jsonl \
    --config config.yaml \
    --checkpoint <checkpoint> \
    --output pred.lst
```

### Get predictions with evaluation(accuracy, confidence interval)

```bash
python eval.py \
    --input_x data/piqa/valid.jsonl \
    --config config.yaml \
    --checkpoint <checkpoint> \
    --input_y data/piqa/valid-labels.lst \
    --output pred.lst
```

