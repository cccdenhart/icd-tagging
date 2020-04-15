"""Classes and functions for modeling."""
import os
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Union, Dict
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gensim.models.keyedvectors import Word2VecKeyedVectors
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from utils import Batcher, ICDDataset


@dataclass
class Lstm(nn.Module):
    """An LSTM implementation with sklearn-like methods."""

    # instance variables
    weights: torch.tensor
    n_code: int
    embedding_dim: int = 300
    lstm_size: int = 128
    batch_size: int = 64
    n_epochs: int = 30

    def __post_init__(self):
        super(Lstm, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(self.weights)
        self.lstm = nn.LSTM(self.embedding_dim, self.lstm_size)
        self.hidden2code = nn.Linear(self.lstm_size, self.n_code)

    def forward(self, X: List[List[int]]) -> torch.tensor:
        # zero pad sequences such that all are length of longest seq
        seq_lens = torch.tensor([len(seq) for seq in X])
        X = [torch.tensor(samp) for samp in X]
        pad_X = pad_sequence(X)

        # get embeddings
        embeds = self.embeddings(pad_X)

        # pack padded sequences
        pack_X = pack_padded_sequence(embeds, seq_lens, enforce_sorted=False)

        # propagate through network
        _, (h_n, _) = self.lstm(pack_X)
        code_space = self.hidden2code(h_n)
        code_scores = torch.sigmoid(code_space).squeeze()
        return code_scores

    def fit(self, X: List[List[int]], Y: List[List[int]]):
        """Train network on training set."""
        # initialize batcher
        dataset = ICDDataset(X, Y)
        batcher = Batcher(dataset, batch_size=self.batch_size)

        # initialize parameters
        self.train()  # set model to train mode
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

        # propagate through network
        print("Training LSTM .....")
        for i in range(self.n_epochs):
            print(f"\tEpoch {i}:", end=" ")
            for X_batch, Y_batch in batcher:

                # zero the parameter gradients
                self.zero_grad()

                # retrieve outputs
                outputs = self.forward(X_batch)

                # determine loss and backprop
                loss = loss_fn(outputs, Y_batch)
                loss.backward()  # calculate gradients
                optimizer.step()  # update parameters

            print(f"loss = {loss}")
        print("done.")
        return self

    def predict(self, X: List[List[int]], threshold: float = 0.5) -> np.ndarray:
        """Give predictions for the given data."""
        probs = self(X)
        pos = torch.where(probs < threshold, probs, torch.ones(*probs.shape))
        neg = torch.where(pos > threshold, pos, torch.zeros(*probs.shape))
        preds = neg.long().numpy()
        return preds

    def predict_proba(self, X: List[List[int]]) -> np.ndarray:
        """Get probabilities for the given data."""
        return self(X).detach().numpy()


@dataclass
class Clf:
    """A wrapper for classifiers allowing for abstract usage in evaluation."""

    # instance variables
    model: Union[BaseEstimator, Lstm]
    name: str
    tokens: List[List[str]]
    roots: List[List[str]]
    embeddings: Dict[str, np.ndarray]  # TODO: fix type

    def __post_init__(self) -> None:
        """Initialize variables to be set."""
        self.X, self.Y, self.class_names = self.build_data()
        self.X_train, self.X_test, self.Y_train, self.Y_test = self.split_data()
        self.preds = []
        self.probs = []

    def __str__(self) -> str:
        """String representation of this clf."""
        return self.name

    def build_data(self) -> Tuple[List[List[Union[int, float]]],
                                  List[List[int]],
                                  List[str]]:
        """Prep data for modeling."""
        # build Y
        mlb = MultiLabelBinarizer()
        Y = mlb.fit_transform(self.roots)
        class_names = mlb.classes_

        # build X
        if isinstance(self.model(BaseEstimator)):
            note_embs = [[self.embeddings[t] for t in note
                          if t in self.embeddings]
                         for note in self.tokens]
            X = [list(np.mean(note, axis=0)) for note in note_embs]
        elif isinstance(self.model(Lstm)):
            X = [[self.embeddings.vocab[tok].index for tok in note
                       if tok in self.embeddings]
                      for note in self.tokens]
        else:
            raise ValueError(f"Model not supported: {str(self.model)}.")

        return X, Y, class_names

    def split_data(self) -> Tuple:
        """Split data for train/test."""
        return train_test_split(self.X, self.Y, test_size=0.3)

    def set_fit(self):
        """Store the fitted model."""
        self.model = self.model.fit(self.X_train, self.Y_train)
        return self

    def set_preds(self):
        """Store predictions."""
        self.preds = self.model.predict(self.X_test)
        return self

    def set_probs(self):
        """Store predicted probabilities."""
        self.probs = self.model.predict_proba(self.X_test)
        return self

    def save(self, loc: str) -> str:
        """Pickles this classifier."""
        fn = str(self) + ".clf"
        fp = os.path.join(loc, fn)
        pickle.dump(self, fp)
        return fp


def train_models(roots: List[List[int]],
                 tokens: List[List[str]],
                 w2v: Word2VecKeyedVectors,
                 is_bl: bool) -> List[Clf]:
    """Train all baseline models and return them in Clf form."""
    # initialize models
    if is_bl:
        models = {
            "LogisticRegression": OneVsRestClassifier(
                LogisticRegression(multi_class="ovr")),
            "RandomForest": OneVsRestClassifier(
                RandomForestClassifier(n_estimators=150,
                                       criterion="entropy")),
            "MLP": MLPClassifier(hidden_layer_sizes=(40, 30), 
                                 learning_rate_init=0.1, 
                                 activation='relu',
                                 solver='adam',
                                 max_iter=200)
        }
    else:
        weights = torch.tensor(w2v.vectors)
        n_codes = len(roots[0])
        models = {"Lstm": Lstm(weights, n_codes)}

    # convert to Clf form
    clfs = [Clf(model, name, tokens, roots, w2v) for name, model in models.items()]

    # train clfs
    trained_clfs = [clf.set_fit() for clf in clfs]

    return trained_clfs
