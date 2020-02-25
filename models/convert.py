"""Write sample data to disk."""
import json
import os
import pickle
from functools import reduce
from typing import List

import pandas as pd

from .constants import CONN, PROJ_DIR
from .preprocess import get_categories


def cat_query(cat: str) -> str:
    """Form a query given a category."""
    query: str = f"select subject_id, text, category from mimiciii.noteevents where category='{cat}' limit 100;"
    return query


def main() -> None:
    """Execute program."""
    # load necessary variables
    print("Reading variables .....")
    cats: List[str] = get_categories(CONN)
    df = pd.read_sql(cat_query(cats[0]), CONN)
    for cat in cats[1:]:
        df = df.append(pd.read_sql(cat_query(cat), CONN))
    df.index = list(range(df.shape[0]))

    # write to disk
    print("Writing to disk .....")
    data_fp = os.path.join(PROJ_DIR, "data", "data.pd")
    data_f = open(data_fp, 'wb')
    pickle.dump(df, data_f)

    print("Done!")
