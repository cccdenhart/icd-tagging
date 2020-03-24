"""Convert raw mimic data to preprocessed features/labels."""
import functools as ft
import os
import re

import pandas as pd
from constants import ICD_COLNAME
from constants import LABEL_FN
from constants import NOTE_COLNAME
from constants import PROJ_DIR
from constants import TFIDF_FN
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pyathena.util import as_pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Callable
from typing import List
from typing import Optional
from typing import Set
from typing import Iterable
from utils import get_conn
from pyathena.connection import Connection as AthenaConn
from icd9.icd9 import Node as ICDNode
from icd9.icd9 import ICD9

# define a query for retrieving notes labeled with icd codes
NOTE_ICD_QUERY: str = f"""
SELECT
  ARRAY_AGG(procedures_icd.icd9_code) as {ICD_COLNAME},
  ARRAY_AGG(noteevents.text)[1] as {NOTE_COLNAME}
FROM
  noteevents
INNER JOIN
  procedures_icd
ON
  noteevents.hadm_id = procedures_icd.hadm_id
GROUP BY
  noteevents.row_id
"""


def read_icd_notes(conn_func: Callable[[], AthenaConn],
                   query: str, limit: Optional[int] = None) -> pd.DataFrame:
    """Read in clinical notes joined on icd codes."""
    # define query string
    full_query: str = query + (f"LIMIT {limit};" if limit else ";")

    # retrieve notes from AWS Athena
    with conn_func() as conn:
        cursor = conn.cursor()
        df: pd.DataFrame = as_pandas(cursor.execute(full_query))
    return df


def get_roots(codes: Iterable[str], tree: ICD9) -> List[ICDNode]:
    """Extract the root of each ICD code."""
    def code_to_node(code: str, n: int) -> ICDNode:
        print(f"{n}/{len(codes)}:", code)
        """Convert the mimic code to a ICD Node."""
        idx = 3
        norm_code = code[:idx] + "." + code[idx:] if len(code) > idx else code
        node = tree.find(norm_code)
        return node

    def icd_to_root(code: ICDNode) -> ICDNode:
        """Get the root of the given code."""
        if not code:
            return None
        if type(code.parent) == ICD9:
            return code
        else:
            return icd_to_root(code.parent)
    roots = [icd_to_root(code_to_node(c, i)) for i, c in enumerate(codes)
             if type(c) == str]
    return roots


def one_hot_labels(labels: List[List[str]]) -> List[List[int]]:
    """Convert the list of ICD code labels to one-hot encodings."""
    # get total distinct labels
    dist_labels: Set[str] = set(ft.reduce(lambda acc, l: acc + l, labels))

    # convert labels to one hot encodings
    one_hot: List[List[str]] = [[1 if d in labs else 0 for d in dist_labels]
                                for labs in labels]
    return one_hot


def process_all_notes(docs: List[str]) -> List[List[str]]:
    """Preprocess notes for embedding."""
    def process_note(doc: str) -> List[str]:
        """Process a single note."""
        # remove anonymized references (ex. "[** ... **]")
        redoc: str = re.sub("\B\[\*\*[^\*\]]*\*\*\]\B", "", doc)

        # replace ICD codes?
        # tokenize and remove stop words
        all_stops = set(stopwords.words("english"))
        toks: List[str] = [w for w in word_tokenize(redoc)
                           if w not in all_stops]
        return toks

    all_toks: List[List[str]] = [process_note(doc) for doc in docs]
    return all_toks


def to_tfidf(docs: List[str]) -> List[List[int]]:
    """Convert the notes to tfidf vectors."""
    # fit notes to sklearn tfidf vectorizer
    vectorizer = TfidfVectorizer().fit(docs)

    # transform notes with fitted vectorizer
    tfidf_mat = vectorizer.transform(docs)

    # convert tfidf from sparse matrix to list representation
    tfidf_list: List[List[int]] = tfidf_mat.todense().tolist()
    return tfidf_list


def main() -> None:
    """Cache preprocessed data files for modeling."""
    # read icd codes and notes
    print("Reading in data .....")
    df = read_icd_notes(get_conn, NOTE_ICD_QUERY, limit=1000)

    # retrieve labels
    print("Retrieving one-hot labels .....")
    one_hots = one_hot_labels(df[ICD_COLNAME].tolist())

    # preproces notes
    print("Preprocessing notes .....")
    pp_notes = process_all_notes(df[NOTE_COLNAME].tolist())

    # retrieve tfidf vectors
    print("Embedding with tfidf .....")
    joined_pp = [" ".join(pp_note) for pp_note in pp_notes]
    tfidfs = to_tfidf(joined_pp)

    # write to disk
    print("Writing to disk .....")
    labels_fp = os.path.join(PROJ_DIR, "data", LABEL_FN)
    pd.DataFrame(one_hots).to_csv(labels_fp)

    tfidf_fp = os.path.join(PROJ_DIR, "data", TFIDF_FN)
    pd.DataFrame(tfidfs).to_csv(tfidf_fp)


if __name__ == "__main__":
    main()
