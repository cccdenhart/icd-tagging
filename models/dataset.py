"""Define a pytorch dataset for a clinical notes model."""
from copy import deepcopy
from random import sample
from typing import List, Sequence, Tuple

from pyathena.connection import Connection
from torch import Tensor, tensor
from torch.utils.data import Dataset

from .preprocess import (cat_data, clean_notes, cust_cossim, cust_led,
                         dim_reduce, get_categories, get_topics,
                         note_similarity, tfidf_diff, vectorize_notes)


class InfoScoreDataset(Dataset):
    """A dataset object for redundancy labeling and feature generation."""

    def __init__(self, conn: Connection, limit: int = 0, n_topics: int = 5) -> None:
        """Initialize class variables."""
        self.conn: Connection = conn
        self.all_ids: List[List[int]] = []
        self.all_notes: List[List[str]] = []
        self.features: Tensor = Tensor()
        self.labels: List[float] = []
        self.limit: int = limit
        self.n_topics: int = n_topics

    def __len__(self) -> int:
        """Return the length of this dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Tensor, float]:
        """Get the features and label at the given index."""
        X: Tensor = self.features[idx]
        y: float = self.labels[idx]
        return X, y

    def read_data(self) -> None:
        """Read data from the connection into the object."""
        print("Reading data .....")
        categories = get_categories(self.conn)
        for cat in categories:
            print(f"\tCategory: {cat} .....", end=" ")
            ids, notes = cat_data(cat, self.conn, limit=self.limit)
            notes = clean_notes(notes)
            self.all_ids.append(ids)
            self.all_notes.append(notes)
            print("done.")
        print("done.")

    def build_labels(self, a: float = 1.0, b: float = 1.0) -> None:
        """
        Generate labels by piping through the steps below.

        :param a: the weight to be applied to cosine similarities
        :param b: the weight to be applied to levenshtein distances

        1. Retrieve categories
        2. Retrieve notes per category
        3. Get Tfidf features per category
        4. Get topics per category
        5. Get cosine similarity between sequential notes
        6. Get levenshtein similarity between seq notes
        7. Log and normalize levenshtein values
        8. Combine similarities through weighting
        """
        print("Building labels ......")
        i: int = 0
        for notes, ids in zip(self.all_notes, self.all_ids):
            # generate topics
            vect_notes: List[List[float]] = vectorize_notes(notes, i)
            topics: List[List[float]] = get_topics(
                vect_notes, i, n_topics=self.n_topics)

            # get cosine similarities
            cos_sims: Sequence[float] = note_similarity(
                topics, ids, cust_cossim)

            # get levenshtein distances, log, then normalize
            led_sims: List[float] = note_similarity(notes, ids, cust_led)

            # compute labels
            y = [(a * cos + b * led) / (a + b)
                 for cos, led in zip(cos_sims, led_sims)]
            self.labels += y
            i += 1
        print("done.")

    def build_features(self) -> None:
        """Generate features through note tfidf differences."""
        print("Building features ......")
        all_features: List[List[float]] = []
        i: int = 0
        for notes, ids in zip(self.all_notes, self.all_ids):
            # get tfidf differences
            vect_notes: List[List[float]] = vectorize_notes(notes, i)
            diffs = tfidf_diff(vect_notes, ids)

            # reduce dimensionality via pca
            # n_feats: int = int(sqrt(len(vect_notes[0])))
            # n_comps: int = n_feats if n_feats < len(vect_notes) else len(vect_notes)
            n_comps = 25
            pca = dim_reduce(diffs, n_comps, i)

            # add to feature set
            all_features += pca
            i += 1
        self.features = tensor(all_features).float()
        print("done.")

    def duplicate(self) -> Dataset:
        """Return a copy of this dataset."""
        dataset = InfoScoreDataset(
            self.conn, limit=self.limit, n_topics=self.n_topics)
        dataset.all_ids = self.all_ids
        dataset.all_notes = self.all_notes
        dataset.features = self.features
        dataset.labels = self.labels
        return dataset

    def train_test_split(self, pct_train: float = 0.7) -> Tuple[Dataset, Dataset]:
        """Split this object into two random subset objects."""
        # get random indexes for features/labels
        train_idxs: List[int] = sample(
            range(len(self)), int(len(self) * pct_train))
        test_idxs: List[int] = [i for i in range(
            len(self)) if i not in train_idxs]

        # copy this features/labels
        feats = deepcopy(self.features).tolist()
        labels = deepcopy(self.labels)

        # get train/test features and labels
        train_feats = tensor([feats[i] for i in train_idxs]).float()
        train_labels = [labels[i] for i in train_idxs]
        test_feats = tensor([feats[i] for i in test_idxs]).float()
        test_labels = [labels[i] for i in test_idxs]

        # assign new feats/labels to copies of this dataset
        trainset = self.duplicate()
        testset = self.duplicate()
        trainset.features = train_feats
        trainset.labels = train_labels
        testset.features = test_feats
        testset.labels = test_labels

        return trainset, testset

    def remove_data(self) -> None:
        """Clear all data in this dataset."""
        self.all_ids = []
        self.all_notes = []
        self.features = Tensor()
        self.labels = []

    def clear_dataset(self) -> None:
        """Clear all labels and features derived in this dataset."""
        self.features = Tensor()
        self.labels = []
