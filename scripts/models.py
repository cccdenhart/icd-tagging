"""Classes and functions for modeling."""
import os
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Union, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gensim.models.keyedvectors import Word2VecKeyedVectors
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from joblib import dump 

from utils import Batcher, ICDDataset


class Lstm(nn.Module):
    """An LSTM implementation with sklearn-like methods."""

    def __init__(self, weights):
        super(Lstm, self).__init__()
        # instance variables
        self.weights: torch.tensor = weights
        self.n_code: int = 15
        self.embedding_dim: int = 300
        self.lstm_size: int = 128
        self.batch_size: int = 64
        self.n_epochs: int = 30
        self.embeddings = nn.Embedding.from_pretrained(self.weights)
        self.lstm = nn.LSTM(self.embedding_dim, self.lstm_size)
        self.hidden2code = nn.Linear(self.lstm_size, self.n_code)

    def forward(self, X: List[List[int]]) -> torch.tensor:
        # zero pad sequences such that all are length of longest seq
        seq_lens = torch.Tensor([len(seq) for seq in X])
        X = [torch.LongTensor(samp) for samp in X]
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

    def __post_init__(self) -> None:
        """Initialize variables to be set."""
        self.preds = []
        self.probs = []

    def __str__(self) -> str:
        """String representation of this clf."""
        return self.name

    def set_fit(self, X, Y):
        """Store the fitted model."""
        print(f"Training {self.name} .....")
        self.model = self.model.fit(X, Y)
        return self

    def set_preds(self, X: List[List[Union[int, float]]]) -> List[List[int]]:
        """Store predictions."""
        self.preds = self.model.predict(X)
        return self.preds

    def set_probs(self, X: List[List[Union[int, float]]]) -> List[List[float]]:
        """Store predicted probabilities."""
        self.probs = self.model.predict_proba(X)
        return self

    def save(self, loc: str) -> str:
        """Pickles this classifier."""
        # deserialize model if lstm
        if isinstance(self.model, Lstm):
            fn = str(self) + ".pt"
            fp = os.path.join(loc, fn)
            torch.save(self.model.state_dict(), fp)
        else:
            fn = str(self) + ".sk"
            fp = os.path.join(loc, fn)
            dump(self.model, fp)
        return fp


def train_baseline(X: List[List[float]],
                   Y: List[List[str]]) -> List[Clf]:
    """Train all baseline models and return them in Clf form."""
    # initialize models
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

    # convert to Clf form
    clfs = [Clf(model, name) for name, model in models.items()]

    # train clfs
    trained_clfs = [clf.set_fit(X, Y) for clf in clfs]

    return trained_clfs


def train_lstm(X, Y, embeddings) -> List[Clf]:
    """Train an lstm model."""
    # instantiate model
    weights = torch.Tensor(embeddings)
    models = {"Lstm": Lstm(weights)}

    # convert to Clf form
    clfs = [Clf(model, name) for name, model in models.items()]

    # train clfs
    trained_clfs = [clf.set_fit(X, Y) for clf in clfs]

    return trained_clfs

