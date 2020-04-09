"""Convert raw mimic data to preprocessed features/labels."""
import functools as ft
import os
import sys
import re
from typing import Callable
from typing import List
from typing import Optional
from typing import Dict
from typing import Set

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from pyathena.connection import Connection as AthenaConn
from pyathena.util import as_pandas

from utils import PROJ_DIR
from utils import TREE
from icd9.icd9 import ICD9
from icd9.icd9 import Node as ICDNode
from utils import get_conn
nltk.download('stopwords')
nltk.download('punkt')


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


def clean_code(code: str, tree: ICD9) -> ICDNode:
    """Convert the mimic code to a ICD Node."""
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
                    return "V01-V91"
                else:
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
    icd_df["icd"] = icd_df["raw_code"].apply(lambda x: clean_code(x, tree))

    # generate the root map
    dist_codes = set(icd_df["icd"].tolist())
    root_map = codes_to_roots(dist_codes, tree)

    # convert codes to roots
    icd_df["roots"] = icd_df["icd"].apply(lambda x: root_map[x])

    df = icd_df.drop(["icd", "raw_code"], axis=1)
    return df


def process_note(doc: str) -> List[str]:
    """Process a single note."""
    # remove anonymized references (ex. "[** ... **]") and lower case
    redoc: str = re.sub(r"\B\[\*\*[^\*\]]*\*\*\]\B", "", doc).tolower()

    # tokenize and remove stop words
    all_stops = set(stopwords.words("english"))
    toks: List[str] = [w for w in word_tokenize(redoc)
                       if w not in all_stops]
    return toks


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
    note_df["tokens"] = note_df["text"].apply(process_note)

    df = note_df.drop("text", axis=1)
    return df


def main() -> None:
    """Cache preprocessed data files for modeling."""

    # define main constants
    subdir = "full_data"

    if "--roots" in sys.argv:
        # process icd codes
        roots_fp = os.path.join(PROJ_DIR, subdir, "roots.csv")
        icd_df = retrieve_icd(get_conn, TREE)
        icd_df.to_csv(roots_fp, index=False)

    if "--notes" in sys.argv:
        # process notes
        notes_fp = os.path.join(PROJ_DIR, subdir, "notes.csv")
        notes_df = retrieve_notes(get_conn)
        notes_df.to_csv(notes_fp, index=False)


if __name__ == "__main__":
    main()
