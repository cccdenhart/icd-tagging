import sys
import os

from utils import PROJ_DIR, TREE, get_conn
from scripts.preprocess import retrieve_icd, retrieve_notes
import pandas as pd
from scripts.embed import retrieve_w2v


def main() -> None:
    """Cache preprocessed data files for modeling."""

    # define main constants
    subdir = "data"
    roots_fp = os.path.join(PROJ_DIR, subdir, "roots.csv")
    notes_fp = os.path.join(PROJ_DIR, subdir, "notes.csv")
    w2v_fp = os.path.join(PROJ_DIR, subdir, "embeddings.csv")

    if "--roots" in sys.argv:
        # process icd codes
        icd_df = retrieve_icd(get_conn, TREE)
        icd_df.to_csv(roots_fp, index=False)

    if "--notes" in sys.argv:
        # process notes
        notes_df = retrieve_notes(get_conn)
        notes_df.to_csv(notes_fp, index=False)

    if "--word2vec" in sys.argv:
        # embed with word2vec
        tok_df = pd.read_csv(notes_fp)
        w2v_df = retrieve_w2v(tok_df)
        w2v_df.to_csv(w2v_fp, index=False)

if __name__ == "__main__":
    main()
