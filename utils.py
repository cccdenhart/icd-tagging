import functools as ft
import os

import pandas as pd
import pyathena
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from typing import List
from typing import Set
from constants import PROJ_DIR, ENV_NAME
from dotenv import load_dotenv
from icd9.icd9 import ICD9
from icd9.icd9 import Node as ICDNode
import networkx as nx


def get_conn():
    """Allow lazy calling of pyathena connection."""
    # load the environment
    load_dotenv(dotenv_path=os.path.join(PROJ_DIR, ENV_NAME))
    return pyathena.connect(aws_access_key_id=os.getenv("ACCESS_KEY"),
                            aws_secret_access_key=os.getenv("SECRET_KEY"),
                            s3_staging_dir=os.getenv("S3_DIR"),
                            region_name=os.getenv("REGION_NAME"))


def process_note(text: str) -> List[str]:
    """Processes a given note."""
    cust_stops = {"[", "]", "(", ")"}
    stop_words = set(stopwords.words('english')) | cust_stops
    toks: List[str] = word_tokenize(text)
    filt_toks = [w for w in toks if w not in stop_words]
    return filt_toks


def score_redund(record: List[List[str]], note_idx: int) -> float:
    """
    Calculate pct text overlap between the given note and its prev record.

    :param record: tokenized patient record
    :param note_idx: the index of the note to score redundancy for

    :returns score: the redundancy score
    """
    note: List[str] = record[note_idx]
    comb_prev: List[str] = ft.reduce(lambda acc, n: acc + n,
                                     record[:note_idx], [])
    dups: int = sum([1 for w in note if w in comb_prev])
    total: int = len(note)
    score: float = 0 if total == 0 else dups / total
    return score


def all_redund(all_records: pd.DataFrame) -> List[float]:
    """
    Calculate a redundancy score for every note.

    NOTE: assumes the given df is sorted ASCENDING by 'charttime'
    """
    records: Set[int] = set(all_records["subject_id"])
    scores: List[float] = []
    for record in tqdm(records):
        rec_df = all_records.loc[all_records["subject_id"] == record, :]
        for i in range(rec_df.shape[0]):
            scores.append(score_redund(rec_df["processed"].tolist(), i))
    return scores


def tree_to_nx(root: ICD9) -> nx.Graph:
    """Convert a ICD9 tree to a networkx graph."""
    G: nx.Graph = nx.Graph()
    stack: List[ICDNode] = root.children.copy()
    while len(stack) > 0:
        node: ICDNode = stack.pop()
        G.add_node(node.code)
        for child in node.children:
            stack.append(child)
            G.add_edge(node.code, child.code)
    return G
