import os
from dataclasses import dataclass
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
import pyathena
from pyathena.connection import Connection as AthenaConn
from pyathena.util import as_pandas
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from icd9.icd9 import ICD9
from icd9.icd9 import Node as ICDNode

""" --- Utility constants --- """
# define project location
PROJ_DIR: str = os.path.dirname(os.path.abspath(__file__))

# load the icd9 tree
TREE = ICD9(os.path.join(PROJ_DIR, "icd9", "codes.json"))
ROOT_DESCS = {n.code: n.description for n in TREE.children}


""" --- Utility objects --- """
@dataclass
class ICDDataset(Dataset):
    """Implementation of PyTorch dataset."""
    # initialize variables
    X: List[List[int]]
    Y: List[List[int]]

    def __len__(self) -> int:
        """Return the length of this dataset."""
        return len(self.Y)

    def __getitem__(self, idx: int) -> Tuple[List[int], torch.tensor]:
        """Get the features and label at the given index."""
        return self.X[idx], torch.FloatTensor(self.Y[idx])


@dataclass
class Batcher():
    """Allow batching of data."""
    # initialize variables
    dataset: Dataset
    batch_size: int = 64
    cur_idx: int = 0

    def __iter__(self) -> None:
        self.cur_idx = 0
        return self

    def __next__(self) -> Tuple[List[List[int]],
                                torch.tensor]:
        """Return the next batch of data."""
        # check if finished iterating
        if self.cur_idx > len(self.dataset):
            raise StopIteration

        # retrieve batch
        end_idx = self.cur_idx + self.batch_size
        if end_idx < len(self.dataset):
            X, Y = self.dataset[self.cur_idx:end_idx]
        else:
            X, Y = self.dataset[self.cur_idx:]

        # increment current index
        self.cur_idx += self.batch_size

        return X, Y


""" --- Utility functions --- """
def get_conn():
    """Allow lazy calling of pyathena connection."""
    # load the environment
    env_name = ".env"
    load_dotenv(dotenv_path=os.path.join(PROJ_DIR, env_name))
    return pyathena.connect(aws_access_key_id=os.getenv("ACCESS_KEY"),
                            aws_secret_access_key=os.getenv("SECRET_KEY"),
                            s3_staging_dir=os.getenv("S3_DIR"),
                            region_name=os.getenv("REGION_NAME"))


def probs_to_preds(probs, threshold):
    """Classifies probabilities with the theshold."""
    return [[0 if p < threshold else 1 for p in row]
            for row in probs]


def ml_accuracy(Y_true: List[List[int]], Y_pred: List[List[int]]) -> float:
    """Calculate multi-label accuracy."""
    ratios = []
    for z, y in zip(Y_true, Y_pred):
        iz = [i for i, val in enumerate(z) if val]
        iy = [i for i, val in enumerate(y) if val]
        sz, sy = set(iz), set(iy)
        ratio = len(sz & sy) / len(sz | sy)
        ratios.append(ratio)
    acc = sum(ratios) / len(ratios)
    return acc


def query_aws(conn_fn: AthenaConn, query: str) -> pd.DataFrame:
    """Execute a query on the Athena database and return as pandas."""
    with conn_fn() as conn:
        cursor = conn.cursor()
        df = as_pandas(cursor.execute(query))
    return df
