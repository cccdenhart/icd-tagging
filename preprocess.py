"""Convert raw mimic data to preprocessed features/labels."""
import functools as ft
import os
import re
from typing import Callable
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pyathena.connection import Connection as AthenaConn
from pyathena.util import as_pandas
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from constants import ICD_COLNAME
from constants import LABEL_FN
from constants import NOTE_COLNAME
from constants import PROJ_DIR
from constants import TFIDF_FN
from constants import TREE
from icd9.icd9 import ICD9
from icd9.icd9 import Node as ICDNode
from utils import get_conn

# define a query for retrieving notes labeled with icd codes
NOTE_ICD_QUERY: str = f"""
SELECT
  ARRAY_AGG(icdtab.icd9_code) as {ICD_COLNAME},
  ARRAY_AGG(notetab.text)[1] as {NOTE_COLNAME}
FROM
  mimiciii.noteevents as notetab
INNER JOIN
  mimiciii.diagnoses_icd as icdtab
ON
  notetab.hadm_id = icdtab.hadm_id
GROUP BY
  notetab.row_id
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
    return df.dropna()


def get_roots(codes: List[str], tree: ICD9) -> List[ICDNode]:
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
        if type(code.parent) == ICD9:
            return code
        else:
            return icd_to_root(code.parent)
    roots = [icd_to_root(code_to_node(c, i)) for i, c in enumerate(codes)]
    return roots


def process_all_notes(docs: List[str]) -> List[List[str]]:
    """Preprocess notes for embedding."""
    def process_note(doc: str) -> List[str]:
        """Process a single note."""
        # remove anonymized references (ex. "[** ... **]")
        redoc: str = re.sub(r"\B\[\*\*[^\*\]]*\*\*\]\B", "", doc)

        # replace ICD codes?
        # tokenize and remove stop words
        all_stops = set(stopwords.words("english"))
        toks: List[str] = [w for w in word_tokenize(redoc)
                           if w not in all_stops]
        return toks

    all_toks: List[List[str]] = [process_note(doc) for doc in docs]
    return all_toks


def to_tfidf(docs: List[str]) -> np.ndarray:
    """Convert the notes to tfidf vectors."""
    # fit notes to sklearn tfidf vectorizer
    vectorizer = TfidfVectorizer().fit(docs)

    # transform notes with fitted vectorizer
    tfidf_mat = vectorizer.transform(docs)

    # convert tfidf from sparse matrix to list representation
    return tfidf_mat.todense()


def to_pca(X: np.ndarray, ndims: int) -> np.ndarray:
    """Reduce the dimensionality of the 2d array to ndims."""
    if ndims <= min([len(cols) for cols in X]):
        pca = PCA(n_components=ndims).fit(X)
        return pca.transform(X)
    else:
        raise AttributeError("Reduced dim size is greater than original dims.")


def main() -> None:
    """Cache preprocessed data files for modeling."""
    # read icd codes and notes
    print("Reading in data .....")
    query_fp = os.path.join(PROJ_DIR, "data", "icd_notes_query.csv")
    if os.path.exists(query_fp):
        df = pd.read_csv(query_fp)
    else:
        df = read_icd_notes(get_conn, NOTE_ICD_QUERY)
        df.to_csv(query_fp, header=True, index=False)

    # extract icd codes
    icd_codes = ft.reduce(lambda acc, codes: acc + codes, df[ICD_COLNAME])

    # read in roots icd codes
    print("Retrieving ICD roots .....")
    roots_fp = os.path.join(PROJ_DIR, "data", "roots_labels.csv")
    if os.path.exists(roots_fp):
        roots_df = pd.read_csv(roots_fp, header=None)
    else:
        dist_icd = list(set(icd_codes))
        root_codes = [r.code for r in get_roots(dist_icd, TREE)]
        root_pairs = {"icd": dist_icd, "root": root_codes}
        roots_df = pd.DataFrame(root_pairs)
        roots_df.to_csv(roots_fp, index=False)

    # generate labels
    roots_map = roots_df.dropna().to_dict('records')
    labels = [roots_map[icd] for icd in icd_codes]
    labels_fp = os.path.join(PROJ_DIR, "data", "labels.csv")
    pd.Series(labels).to_csv(labels_fp, header=False, index=False)

    # preproces notes
    print("Preprocessing notes .....")
    notes_fp = os.path.join(PROJ_DIR, "data", "processed_notes.csv")
    if os.path.exists(notes_fp):
        pp_notes = pd.read_csv(notes_fp, header=None)
    else:
        # preprocess notes
        pp_notes = process_all_notes(df[NOTE_COLNAME].tolist())

        # replicate notes by the number of icd codes associated with it
        rep_pp = ft.reduce(lambda acc, notes: acc + notes,
                           [],
                           map(lambda note, codes: [note] * len(codes),
                               pp_notes,
                               df[ICD_COLNAME].tolist()))
        pd.DataFrame(rep_pp).to_csv(notes_fp, header=False, index=False)

    # retrieve tfidf vectors
    print("Embedding with tfidf .....")
    tfidf_fp = os.path.join(PROJ_DIR, "data", TFIDF_FN)
    if os.path.exists(tfidf_fp):
        tfidfs = pd.read_csv(tfidf_fp, header=None).values()
    else:
        joined_pp = [" ".join(pp_note) for pp_note in pp_notes]
        tfidfs = to_tfidf(joined_pp)
        pd.DataFrame(tfidfs).to_csv(tfidf_fp, header=False, index=False)

    # retrieve PCA values
    print("Reducing tfidf with pca .....")
    pca = to_pca(np.array(tfidfs), 300)
    pca_fp = os.path.join(PROJ_DIR, "data", "pca.csv")
    pd.DataFrame(pca).to_csv(pca_fp, header=False, index=False)


if __name__ == "__main__":
    main()
