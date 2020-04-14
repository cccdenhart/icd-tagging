"""Embed clinical notes."""
from typing import List
import numpy as np
import pandas as pd
from utils import PROJ_DIR
import os
from gensim.models import KeyedVectors


def load_word2vec():
    w2v_fp = os.path.join(PROJ_DIR, "embeddings",
                          "GoogleNews-vectors-negative300.bin")
    word2vec = KeyedVectors.load_word2vec_format(w2v_fp, binary=True)
    return word2vec


def toks2vec(doc: List[str], model) -> List[np.array]:
    """Embed the tokenized document with word2vec."""
    return [model[t] for t in doc if t in model]


def retrieve_w2v(tok_df: pd.DataFrame) -> pd.DataFrame:
    """Embed the tokens in the given df with word2vec."""
    word2vec = load_word2vec()
    tok_df["w2v"] = tok_df["tokens"]\
        .apply(lambda x: toks2vec(x, word2vec))
    return tok_df
