'''
train.py contains methods for training our bert models
'''

from transformers import AutoModelForSequenceClassification
from torch.optim.lr_scheduler import LambdaLR
from sklearn.pipeline import Pipeline
from skorch.hf import HuggingfacePretrainedTokenizer
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, LRScheduler, ProgressBar
from tqdm import tqdm

import model as m 
import torch.nn as nn
import torch
import click
import csv
import sys
import random
import numpy as np
import pickle

csv.field_size_limit(sys.maxsize)


# return a training set sampled from training data as well as its label dictionary
def get_X_y_from_file(filename, label_column=3, label_dict=None, limit_=None):
    X = []
    y = []
    with open(filename, 'r') as f:
        tsv_reader = csv.reader(f, delimiter="\t")
        next(tsv_reader)
        if limit_:
            subsample = sorted(list(tsv_reader), key=lambda k: random.random())[:limit_]
        else:
            subsample = list(tsv_reader)
        for line in tqdm(subsample):
            y.append(line[label_column])
            sent = line[0]
            tokens = sent.split()
            sent = ' '.join(tokens)
            X.append(sent)
    if not label_dict:
        label_dict = dict([(y1, y2) for y2, y1 in enumerate(set(y))])
    y = np.array([label_dict[l] if l in label_dict else '-1' for l in y])
    return X, y, label_dict 


# get appropriate device
def get_device(no_mps=False):
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.has_mps and not no_mps:
        return torch.device("mps")
    else:
        return torch.device("cpu")
    

# save finetuned models' param after training 
def save_model(net, label_dict_inv):
    model = net.steps[1][1].module_.bert
    tokenizer = net.steps[0][1].fast_tokenizer_
    model.save_pretrained('./saved/period/model/')
    tokenizer.save_pretrained('./saved/period/tokenizer/')

    dict_path_author = './saved/period/dict_period.pkl'
    with open(dict_path_author, 'wb') as file:
        pickle.dump(label_dict_inv, file)
    

@click.command()
@click.option("--batch-size", default=32, help='batch size (default 32)')
@click.option("--max-epochs", default=1, help='number of epochs')
@click.option("--predict-level", default='time_period', type=click.Choice(["author", "time_period", "dialect"]), help='the label to predict')
@click.option("--limit", default=None, help='max # of examples to train on')
@click.option("--train-file", type=click.Path(readable=True), default= 'data/preprocessed/all_data_masked.tsv', required=True, help='path for training .tsv file')
@click.option("--dev-file", type=click.Path(readable=True), required=False, help='path for dev/test .tsv file')
@click.option("--log-file", default='log.txt', help='path for .txt file for logging results')
@click.option("--lr", default=0.00005, help='learning rate (default 0.00005)')
@click.option("--model-name", type=click.Choice(["dan", "lstm", "bilstm", "cnn", "bert"]), default='bert', help='model name')
@click.option("--llm-model", type=click.Choice(["bert", "distilbert"]), default="distilbert", help='pretrained model'\
    ' to use if model-name is "bert"')
# the training function 
# Modified from HuggingFace tutorial: 
# https://huggingface.co/scikit-learn/skorch-text-classification/blob/main/train.py
def train_model(batch_size, max_epochs, predict_level, limit,
    train_file, lr, model_name, llm_model,
    ):

    # Define f1 metrics
    f1_micro = EpochScoring("f1_micro")
    f1_macro = EpochScoring("f1_macro")
    
    if model_name == 'bert':
        if predict_level == 'author':
            column = 3
        elif predict_level == 'time_period':
            column = 2
        else: 
            column = 1
        if limit:
            limit = int(limit)

        # X_train, y_train, label_dict = get_X_y_from_file(train_file, limit_=limit, label_column=column)
        X_train, y_train, label_dict = get_X_y_from_file(train_file, label_column=column)
        num_training_steps = max_epochs * (len(X_train) // batch_size + 1)

        label_dict_inv = dict([(y,x) for (x,y) in label_dict.items()])

        def lr_schedule(current_step):
            factor = float(num_training_steps - current_step) / float(max(1, num_training_steps))
            assert factor > 0
            return factor

        # set up training pipeline 
        net = Pipeline([
            ('tokenizer', HuggingfacePretrainedTokenizer(f"{llm_model}-base-uncased")),
            ('net', NeuralNetClassifier(
                module=m.BERTClassifier,
                module__name=f"{llm_model}-base-uncased",
                module__num_labels=len(set(y_train)),
                optimizer=torch.optim.AdamW,
                lr=lr,
                max_epochs=max_epochs,
                criterion=nn.CrossEntropyLoss,
                batch_size=batch_size,
                iterator_train__shuffle=True,
                device=get_device(),
                callbacks=[
                    f1_micro, f1_macro,
                    LRScheduler(LambdaLR, lr_lambda=lr_schedule, step_every='batch'),
                    ProgressBar(),
                ],
            )),
        ])

    # train the model
    net.fit(X_train, y_train)
    return net, label_dict_inv


# given a test tsv, return input X and gold y 
def x_y(tsv_file):
    text_values = []
    label_values = []
    
    with open(tsv_file, 'r', newline='') as file:
        reader = csv.DictReader(file, delimiter='\t')
        next(reader)
        for row in reader:
            text_values.append(row['paragraph'])
            label_values.append(row['period'])
    return text_values, label_values


# a simple accuracy evaluation on trained model on test data 
def evaluate(model, tokenizer, texts, labels, label_dict):
    right = 0 
    for i in range(len(labels)):
        tokens = tokenizer.encode_plus(
            texts[i],
            padding="max_length",
            truncation=True,
            return_tensors="pt"  # Returns PyTorch tensors
            )
        outputs = model(**tokens)
        pred = torch.argmax(outputs.logits)
        prediction = label_dict[int(pred)]
        if prediction == labels[i]:
            right += 1
    
    print('Test accuracy: ', right/len(labels))


if __name__ == '__main__':
    author_model, author_dict = train_model('time_period')
    save_model(author_model, author_dict)