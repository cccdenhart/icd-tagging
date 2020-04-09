import sys
import os

from utils import PROJ_DIR, TREE, get_conn
from scripts.preprocess import retrieve_icd, retrieve_notes


def main() -> None:
    """Cache preprocessed data files for modeling."""

    # define main constants
    subdir = "data"

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
