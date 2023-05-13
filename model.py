'''
Contains model designs
Contains BERT classifiers for authors, dialects, and time periods
'''
import torch.nn as nn
import torch
from transformers import AutoModelForSequenceClassification

# Taken from HuggingFace tutorial on skorch:
# https://huggingface.co/scikit-learn/skorch-text-classification/blob/main/train.py
class BERTClassifier(nn.Module):
    def __init__(self, name, num_labels):
        super().__init__()
        self.name = name
        self.num_labels = num_labels
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            self.name, num_labels=self.num_labels
        )
    def forward(self, **kwargs):
        pred = self.bert(**kwargs)
        return pred.logits
    def get_bert(self):
        return self.bert
