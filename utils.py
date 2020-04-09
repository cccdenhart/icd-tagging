import functools as ft
import os
from typing import List
from typing import Set
import re

import networkx as nx
import pandas as pd
import pyathena
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple

from constants import ENV_NAME
from constants import PROJ_DIR
from icd9.icd9 import ICD9
from icd9.icd9 import Node as ICDNode


class ICDDataset(Dataset):
    """Implementation of PyTorch dataset."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """Initialize class variables."""
        self.X = X
        self.y = y

    def __len__(self) -> int:
        """Return the length of this dataset."""
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor,
                                             torch.FloatTensor]:
        """Get the features and label at the given index."""
        xi: torch.FloatTensor = torch.FloatTensor(self.X[idx])
        yi: torch.FloatTensor = torch.FloatTensor(self.y[idx])
        return xi, yi


def get_conn():
    """Allow lazy calling of pyathena connection."""
    # load the environment
    load_dotenv(dotenv_path=os.path.join(PROJ_DIR, ENV_NAME))
    return pyathena.connect(aws_access_key_id=os.getenv("ACCESS_KEY"),
                            aws_secret_access_key=os.getenv("SECRET_KEY"),
                            s3_staging_dir=os.getenv("S3_DIR"),
                            region_name=os.getenv("REGION_NAME"))


def process_all_notes(docs: List[str]) -> List[List[str]]:
    """Preprocess notes for embedding."""
    def process_note(doc: str) -> List[str]:
        """Process a single note."""
        # remove anonymized references (ex. "[** ... **]")
        redoc: str = re.sub(r"\B\[\*\*[^\*\]]*\*\*\]\B", "", doc)

        # tokenize and remove stop words
        all_stops = set(stopwords.words("english"))
        toks: List[str] = [w for w in word_tokenize(redoc)
                           if w not in all_stops]
        return toks

    all_toks: List[List[str]] = [process_note(doc) for doc in docs]
    return all_toks


def group_v_roots(roots: List[str]) -> List[str]:
    """Maps V ICD codes to a single category."""
    return ["V01-V91" if r[0] == "V" else r
            for r in roots]


def probs_to_preds(probs, threshold):
    """Classifies probabilities with the theshold."""
    return [[0 if p < threshold else 1 for p in row]
            for row in probs]
