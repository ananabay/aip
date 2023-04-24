import model as m

from torch.optim.lr_scheduler import LambdaLR
from sklearn.pipeline import Pipeline
from skorch.hf import HuggingfacePretrainedTokenizer
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, LRScheduler, ProgressBar

import click
import torch.nn as nn
import torch
import csv
import sys
from tqdm import tqdm
import random
import numpy as np

csv.field_size_limit(sys.maxsize)

def get_X_y_from_file(filename, label_dict=None, limit=None):
    X = []
    y = []
    with open(filename, 'r') as f:
        tsv_reader = csv.reader(f, delimiter="\t")
        if limit:
            subsample = sorted(list(tsv_reader), key=lambda k: random.random())[:limit]
        else:
            subsample = list(tsv_reader)
        for line in tqdm(subsample):
            y.append(line[3])
            sent = line[0]
            tokens = sent.split()
            sent = ' '.join(tokens)
            X.append(sent)
    if not label_dict:
        label_dict = dict([(y1, y2) for y2, y1 in enumerate(set(y))])
    y = np.array([label_dict[l] if l in label_dict else '-1' for l in y])
    return X, y, label_dict 


@click.command()
@click.option("--batch-size", default=32, help='batch size (default 32)')
@click.option("--max-epochs", default=10, help='number of epochs')
@click.option("--train-file", type=click.Path(readable=True), required=True, help='path for training .tsv file')
@click.option("--dev-file", type=click.Path(readable=True), required=False, help='path for dev/test .tsv file')
@click.option("--log-file", default='log.txt', help='path for .txt file for logging results')
@click.option("--lr", default=0.00005, help='learning rate (default 0.00005)')
@click.option("--model-name", type=click.Choice(["dan", "lstm", "bilstm", "cnn", "bert"]), default='bert', help='model name')
@click.option("--llm-model", type=click.Choice(["bert", "distilbert"]), default="distilbert", help='pretrained model'\
    ' to use if model-name is "bert"')
def main(batch_size, max_epochs,
    train_file, dev_file, log_file,
    lr, model_name, llm_model,
):

    # Define some metrics
    f1_micro = EpochScoring("f1_micro")
    f1_macro = EpochScoring("f1_macro")

    # Modified from HuggingFace tutorial: 
    # https://huggingface.co/scikit-learn/skorch-text-classification/blob/main/train.py
    if model_name == 'bert':

        X_train, y_train, label_dict = get_X_y_from_file(train_file, limit=1000)
        num_training_steps = max_epochs * (len(X_train) // batch_size + 1)

        def lr_schedule(current_step):
            factor = float(num_training_steps - current_step) / float(max(1, num_training_steps))
            assert factor > 0
            return factor

        net = Pipeline([
            ('tokenizer', HuggingfacePretrainedTokenizer(f"{llm_model}-base-uncased")),
            ('net', NeuralNetClassifier(
                m.BERTClassifier,
                module__name=f"{llm_model}-base-uncased",
                module__num_labels=len(set(y_train)),
                optimizer=torch.optim.AdamW,
                lr=lr,
                max_epochs=max_epochs,
                criterion=nn.CrossEntropyLoss,
                batch_size=batch_size,
                iterator_train__shuffle=True,
                callbacks=[
                    f1_micro, f1_macro,
                    LRScheduler(LambdaLR, lr_lambda=lr_schedule, step_every='batch'),
                    ProgressBar(),
                ],
            )),
        ])

    # train the model
    net.fit(X_train, y_train)

if __name__ == "__main__":
    main()
