"""Embed clinical notes."""
from typing import List
import numpy as np
import pandas as pd
from utils import PROJ_DIR
import os
import sys
import gensim
from gensim.test.utils import datapath
from gensim.models import KeyedVectors


def load_word2vec():
    w2v_fp = os.path.join(PROJ_DIR, "embeddings",
                          "GoogleNews-vectors-negative300.bin")
    word2vec = KeyedVectors.load_word2vec_format(w2v_fp, binary=True)
    return word2vec


def toks2vec(doc: List[str], model) -> List[np.array]:
    """Embed the tokenized document with word2vec."""
    return [model[t] for t in doc if t in model]


def main():
    """Generate embeddings and store."""
    # constants
    data_dir = "full_data"

    # read in tokens df
    tokens_fp = os.path.join(PROJ_DIR, data_dir, "notes.csv")
    tokens_df = pd.read_csv(tokens_fp)

    if "--word2vec" in sys.argv:
        # embed with word2vec
        word2vec = load_word2vec()
        tokens_df["w2v"] = tokens_df["tokens"]\
            .apply(lambda x: toks2vec(x, word2vec))

    if "--bert" in sys.argv:
        # embed with clinical bert
        pass

    # write embeddings to disk
    emb_fp = os.path.join(PROJ_DIR, data_dir, "embeddings.csv")
    df = tokens_df.drop("tokens", axis=1)
    df.to_csv(emb_fp, index=False)


if __name__ == "__main__":
    main()
