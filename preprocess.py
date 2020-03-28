"""Convert raw mimic data to preprocessed features/labels."""
import functools as ft
import os
import re
from typing import Callable
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pyathena.connection import Connection as AthenaConn
from pyathena.util import as_pandas
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from constants import ICD_COLNAME
from constants import NOTE_COLNAME
from constants import PROJ_DIR
from constants import TFIDF_FN
from constants import TREE
from icd9.icd9 import ICD9
from icd9.icd9 import Node as ICDNode
from utils import get_conn
nltk.download('stopwords')
nltk.download('punkt')

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
    full_query: str = query + (f"\nLIMIT {limit};" if limit else ";")

    # retrieve notes from AWS Athena
    with conn_func() as conn:
        cursor = conn.cursor()
        df: pd.DataFrame = as_pandas(cursor.execute(full_query))
    return df.dropna()


def code_to_node(code: str, tree: ICD9,
                 prt_str: Optional[str] = None) -> ICDNode:
    if prt_str:
        print(prt_str)
    """Convert the mimic code to a ICD Node."""
    idx = 3
    norm_code = code[:idx] + "." + code[idx:] if len(code) > idx else code
    node = tree.find(norm_code)
    return node


def icd_to_root(code: ICDNode) -> Optional[ICDNode]:
    """Get the root of the given code."""
    if type(code) != ICDNode and type(code) != ICD9:
        return None
    elif type(code.parent) == ICD9:
        return code
    else:
        return icd_to_root(code.parent)


def get_roots(codes: List[str], tree: ICD9) -> List[Optional[ICDNode]]:
    """Extract the root of each ICD code."""
    roots = [icd_to_root(code_to_node(c, tree, f"Code: {i}/{len(codes)}"))
             for i, c in enumerate(codes)]
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
    return np.array(tfidf_mat.todense())


def to_pca(X: np.ndarray, ndims: int) -> np.ndarray:
    """Reduce the dimensionality of the 2d array to ndims."""
    if ndims <= min([len(cols) for cols in X]):
        pca = PCA(n_components=ndims).fit(X)
        return pca.transform(X)
    else:
        raise AttributeError("Reduced dim size is greater than original dims.")


def extract_icds(code_arrs: List[str]) -> List[str]:
    """Extract icd codes from AWS return."""
    def str_to_list(s: str) -> List[str]:
        """Convert a string rep of a list to a list."""
        return s.strip("[]").split(", ")
    icd_codes = ft.reduce(lambda acc, codes: acc + str_to_list(codes),
                          code_arrs,
                          [])
    return icd_codes

def codes_to_roots(codes: List[str], root_map: Dict[str, str]) -> List[str]:
    """Returns a unique list of roots in the given list of codes."""
    return [root_map[c] if c in root_map.keys() else None for c in codes]


def main() -> None:
    """Cache preprocessed data files for modeling."""
    # define main constants
    subdir = "lim100k_data"

    # process roots labels
    roots_labels_fp = os.path.join(PROJ_DIR, subdir, "roots_labels.csv")
    if not os.path.exists(roots_labels_fp):
        # read icd codes and notes
        print("Reading in data .....")
        df = read_icd_notes(get_conn, NOTE_ICD_QUERY)

        # clean icd codes
        print("Cleaning ICD codes .....")
        CLEAN_COLNAME = "clean_icds"
        df[CLEAN_COLNAME] = df[ICD_COLNAME].apply(lambda x: x.strip("[]")
                                                             .split(", "))

        # extract unique icd codes
        print("Building map from ICD leaves to roots .....")
        icd_codes = list(set(ft.reduce(lambda acc, n: acc + n,
                                       df[CLEAN_COLNAME].tolist(),
                                       [])))
        root_codes = [r.code if r else None
                      for r in get_roots(icd_codes, TREE)]
        root_map = {"icd": icd_codes, "root": root_codes}

        # generate labels
        print("Generating labels .....")
        ROOTS_COLNAME = "root_labels"
        df[ROOTS_COLNAME] = df[CLEAN_COLNAME].apply(codes_to_roots,
                                                    args=(root_map))
        roots_fp = os.path.join(PROJ_DIR, subdir, "labels.csv")
        root_df = df[ROOTS_COLNAME]
        root_df.to_csv(roots_fp)

    # process notes
    doc2vec_fp = os.path.join(PROJ_DIR, subdir, "doc2vec.csv")
    if not os.path.exists(doc2vec_fp):
        # preproces notes
        print("Preprocessing notes .....")
        notes_fp = os.path.join(PROJ_DIR, subdir, "processed_notes.csv")
        pp_notes = process_all_notes(df[NOTE_COLNAME].tolist())

        # replicate notes by the number of icd codes associated with it
        rep_pp = ft.reduce(lambda acc, notes: acc + notes,
                           list(map(lambda note, codes: [note] * len(codes),
                                    pp_notes,
                                    df["clean_icds"].tolist())),
                           [])

        # retrieve tfidf vectors
        print("Embedding with tfidf .....")
        tfidf_fp = os.path.join(PROJ_DIR, subdir, TFIDF_FN)
        if os.path.exists(tfidf_fp):
            tfidfs = pd.read_csv(tfidf_fp, header=None).values
        else:
            joined_pp = [" ".join(pp_note) for pp_note in rep_pp]
            tfidfs = to_tfidf(joined_pp)
            pd.DataFrame(tfidfs).to_csv(tfidf_fp, header=False, index=False)

        # retrieve PCA values
        print("Reducing tfidf with pca .....")
        pca = to_pca(np.array(tfidfs), 50)
        pca_fp = os.path.join(PROJ_DIR, subdir, "pca.csv")
        pca_df = pd.DataFrame(pca)
        pca_df.to_csv(pca_fp, header=False, index=False)

    # write full dataframe
    print("Processing full dataframe .....")
    breakpoint()
    full_df = pca_df.copy()
    full_df["roots"] = labels
    full_df = full_df.dropna()
    full_df_fp = os.path.join(PROJ_DIR, subdir, "full_data.csv")
    full_df.to_csv(full_df_fp, header=False, index=False)


if __name__ == "__main__":
    main()
