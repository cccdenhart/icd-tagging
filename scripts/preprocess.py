"""Convert raw mimic data to preprocessed features/labels."""
import functools as ft
import itertools as it
import os
import re
import sys
from typing import Callable, Dict, List, Optional, Set, Tuple
import math

import nltk
import numpy as np
import pandas as pd
from gensim.models.keyedvectors import Word2VecKeyedVectors
from transformers import AutoTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from pyathena.connection import Connection as AthenaConn
from pyathena.util import as_pandas

from icd9.icd9 import ICD9
from icd9.icd9 import Node as ICDNode
from utils import PROJ_DIR, TREE, get_conn, V_CODE


def read_athena(conn_func: Callable[[], AthenaConn],
                query: str,
                limit: Optional[int] = None) -> pd.DataFrame:
    """Read in clinical notes joined on icd codes."""
    # define query string
    full_query: str = query + (f"\nORDER BY RAND()\nLIMIT {limit};"
                               if limit else ";")

    # retrieve notes from AWS Athena
    with conn_func() as conn:
        cursor = conn.cursor()
        df: pd.DataFrame = as_pandas(cursor.execute(full_query))
    return df.dropna()


def clean_code(code: str) -> str:
    """Format the mimiciii code to standard form."""
    idx = 3
    norm_code = code[:idx] + "." + code[idx:] if len(code) > idx else code
    return norm_code


def codes_to_roots(codes: Set[str], tree: ICD9) -> Dict[str, str]:
    """Extract the root of each ICD code."""
    def icd_to_root(code: str) -> Optional[str]:
        """Get the root of the given code."""
        node = tree.find(code)
        if node:
            parents = node.parents
            if parents and len(parents) > 2:
                root = parents[1].code
                if root[0] == "V":
                    return V_CODE
                return root
        return None
    root_map = {c: icd_to_root(c) for c in codes}
    return root_map


def retrieve_icd(conn_func: Callable[[], AthenaConn],
                 tree: ICD9,
                 limit: Optional[int] = None) -> pd.DataFrame:
    """Retrieves all diagnoses ICD9 codes and paired hadm_id from mimic."""
    query = f"""
    SELECT row_id as icd_id, hadm_id, icd9_code as raw_code
    FROM mimiciii.diagnoses_icd
    """
    # read in the codes and ids
    icd_df = read_athena(conn_func, query, limit)

    # clean the codes
    icd_df["icd"] = icd_df["raw_code"].apply(clean_code)

    # generate the root map
    dist_codes = set(icd_df["icd"].tolist())
    root_map = codes_to_roots(dist_codes, tree)

    # convert codes to roots
    icd_df["roots"] = icd_df["icd"].apply(lambda x: root_map[x])

    df = icd_df.drop(["icd", "raw_code"], axis=1)
    return df


def process_note(doc: str) -> str:
    """Process a single note."""
    # remove anonymized references (ex. "[** ... **]") and lower case
    redoc: str = re.sub(r"\B\[\*\*[^\*\]]*\*\*\]\B", "", doc).lower()
    return redoc


def retrieve_notes(conn_func: Callable[[], AthenaConn],
                   limit: Optional[int] = None) -> pd.DataFrame:
    """Retrieve clinical notes from MIMIC and process."""
    # read in notes
    query = """
    SELECT
      row_id as note_id, hadm_id, text
    FROM
      mimiciii.noteevents
    """
    note_df = read_athena(conn_func, query, limit)

    # clean notes and tokenize
    note_df["text"] = note_df["text"].apply(process_note)

    return note_df


def group_data(roots_df: pd.DataFrame,
               notes_df: pd.DataFrame,
               w2v,
               tokenizer) -> pd.DataFrame:
    """Group the roots and notes data for modeling."""
    # map each note_id to its tokens
    note_map = dict(notes_df.loc[:, ["note_id", "text"]].values)
    hadm_map = dict(notes_df.loc[:, ["note_id", "hadm_id"]].values)

    # join icd roots with notes
    print("Merging note and roots .....")
    df = roots_df.merge(notes_df, on="hadm_id", how="inner").dropna()

    # group by admission
    print("Grouping by hadm id .....")
    df = df.groupby("hadm_id").aggregate(list).reset_index()

    # get unique roots and notes per grouping
    print("Replicating notes .....")
    df["roots"] = df["roots"].apply(lambda x: list(set(x)))
    df["note_id"] = df["note_id"].apply(lambda x: list(set(x)))

    # replicate root lists for each note they are related to
    roots = list(it.chain.from_iterable(map(lambda r, nids: [r] * len(nids),
                                            df["roots"].tolist(),
                                            df["note_id"].tolist())))

    # flatten note ids
    note_ids = list(it.chain.from_iterable(df["note_id"].tolist()))

    # flatten notes grouped by hadm_id
    notes = [note_map[nid] for nid in note_ids]

    # reassign hadm_id for each note id
    hadm_ids = [hadm_map[nid] for nid in note_ids]

    # store the resulting replications in a modeling df
    model_df = pd.DataFrame({"roots": roots, "text": notes, "hadm_id": hadm_ids})

    # tokenize and remove stop words
    print("Creating tokens .....")
    all_stops = set(stopwords.words("english"))
    model_df["tokens"] = model_df["text"]\
        .apply(lambda t: [w for w in word_tokenize(t) if w not in all_stops])

    # remove rows with no tokens from word2vec
    model_df["tokens"] = model_df["tokens"]\
        .apply(lambda x: [t for t in x if t in w2v])
    model_df["tokens"] = model_df["tokens"]\
        .apply(lambda x: None if len(x) == 0 else x)
    model_df = model_df.dropna()

    # average word embeddings to generate d2v embeddings
    print("Creating d2v .....")
    model_df["d2v"] = model_df["tokens"]\
        .apply(lambda doc: list(np.mean([w2v[t] for t in doc if t in w2v],
                                        axis=0)))

    # get column for embedding indices
    print("Creating w2v indices .....")
    model_df["w2v_idx"] = model_df["tokens"]\
        .apply(lambda doc: [w2v.vocab[w].index for w in doc if w in w2v])

    # get bert embeddings indices
    print("Creating bert indices .....")
    model_df["bert_idx"] = model_df["text"]\
        .apply(lambda doc: torch.tensor(tokenizer\
                                        .encode(doc, add_special_tokens=True))\
               .unsqueeze(0))

    # one hot encode labels
    mlb = MultiLabelBinarizer()
    model_df["roots"] = mlb.fit_transform(model_df["roots"])

    return model_df, mlb.classes_


def split_df(df: pd.DataFrame,
             test_size: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the given dataframe in train/test sets."""
    # get the test set
    n_test = math.floor(df.shape[0] * test_size)
    test_df = df.sample(n_test)

    # get the train set
    train_idx = [i for i in df.index if i not in test_df.index]
    train_df = df.loc[train_idx]

    return train_df, test_df

