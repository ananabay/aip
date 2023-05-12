'''
contains model designs
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


class DANClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
        embedding_dim: int,
        hidden_layer_sizes: list[int],
    ):
        super(DANClassifier, self).__init__()

        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.EmbeddingBag(
            num_embeddings=vocab_size, embedding_dim=self.embedding_dim, mode="mean"
        )
        self.fc = nn.Sequential()
        in_out_dims = zip([self.embedding_dim] + hidden_layer_sizes, hidden_layer_sizes[1:])
        for idx, (in_dim, out_dim) in enumerate(in_out_dims):
            self.fc.add_module(
                name=f"{idx}_in{in_dim}_out{out_dim}", module=nn.Linear(in_dim, out_dim)
            )
        self.proj = nn.Linear(hidden_layer_sizes[-1], self.num_classes)

    def forward(self, token_indices: torch.Tensor, *args, **kwargs):
        avg_emb = self.embedding(token_indices)
        out = self.fc(avg_emb)
        out = self.proj(out)
        return out

class LSTMClassifier(nn.Module):

    def __init__(self, emb_input_dim=0, emb_output_dim=0, hidden_size=50, num_classes=2, dr=0.4,
                bidirectional=False, use_embedding=True):
        super(LSTMClassifier, self).__init__()
        self.num_layers = 2
        self.embedding = nn.Embedding(emb_input_dim, emb_output_dim)
        self.lstm = nn.LSTM(num_layers=self.num_layers, input_size=emb_output_dim, hidden_size=hidden_size, dropout=dr,
                            bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=dr)
        if bidirectional:
            self.project = nn.Linear(hidden_size*2, num_classes)
        else:
            self.project = nn.Linear(hidden_size, num_classes)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.apply(self._init_weights)
        self.use_embedding=use_embedding

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight.data)
            module.bias.data.zero_()
        elif isinstance(module, nn.LSTM):
            for i in range(self.num_layers):
                torch.nn.init.xavier_uniform_(module.all_weights[i][0])
                torch.nn.init.xavier_uniform_(module.all_weights[i][1])
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight.data)

    def from_embedding(self, embedded):                             # (batch_size, seq_length, emb_size)
        output = embedded.permute(1, 0, 2)                          # (seq_length, batch_size, emb_size)
        output, _ = self.lstm(output)                               # (seq_length, batch_size, hidden_size)
        output = output.permute(1, 2, 0)                            # (batch_size, hidden_size, seq_length)
        output = self.maxpool(output)                               # (batch_size, hidden_size, 1)
        output = self.dropout(output)                               # (batch_size, hidden_size, 1)
        output = output.squeeze()                                   # (batch_size, hidden_size)
        output = self.project(output)                               # (batch_size, num_classes)
        return output

    def forward(self, data):
        if self.use_embedding:
            data = self.embedding(data)
        return self.from_embedding(data)

class CNNClassifier(nn.Module):

    def __init__(self, emb_input_dim=0, emb_output_dim=0,
                 num_classes=2, use_embedding=True,
                 dr=0.2,
                 filter_widths=[3, 4],
                 num_filters=100,
                 num_conv_layers=1,
                 intermediate_pool_size=3,
                **kwargs):
        super(CNNClassifier, self).__init__(**kwargs)
        self.use_embedding = use_embedding
        self.embedding = (nn.Embedding(emb_input_dim, emb_output_dim) 
                                       if emb_input_dim > 0 
                                       else None)
        self.encoder = []
        for filter_width in filter_widths:
            # CNN block for each filter_width
            seq = nn.Sequential()
            for i in range(num_conv_layers):
                # each block consists of: Dropout, Conv1d, ReLU, pooling
                seq.append(nn.Dropout(p=dr))
                seq.append(nn.Conv1d((num_filters if i > 0 else emb_output_dim),
                                      num_filters, 
                                      kernel_size=filter_width))
                seq.append(nn.ReLU())
                if i == num_conv_layers-1:
                    seq.append(nn.AdaptiveMaxPool1d(1))
                else:
                    seq.append(nn.MaxPool1d(intermediate_pool_size, 1))
            self.encoder.append(seq)
            del seq
        self.encoder = nn.ModuleList(self.encoder)
        # final projection layer
        self.output = nn.Linear(num_filters*len(filter_widths), num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.xavier_uniform_(module.weight.data)

    def from_embedding(self, embedded):
        output = embedded                          # (batch_size, seq_len, emb_output_dim)
        output = output.permute(0,2,1)             # (batch_size, emb_output_dim, seq_len)
        outputs = []
        for layer in self.encoder:
            outputs.append(layer(output))          # (batch_size, num_filters, 1) each
        encoded = torch.cat(outputs, dim=1)        # (batch_size, len(filter_widths)*num_filters, 1)
        encoded = encoded.squeeze()                # (batch_size, len(filter_widths)*num_filters)
        projected = self.output(encoded)           # (batch_size, num_classes = 4)
        del output, outputs, layer, encoded
        return projected

    def forward(self, data):
        if self.use_embedding:
            data = self.embedding(data)
        return self.from_embedding(data)