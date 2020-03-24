"""Functions that generate specific data for visualizations."""
import functools as ft
import re
from collections import Counter
import os

import pandas as pd
from constants import PROJ_DIR, TREE
from icd9.icd9 import ICD9
from icd9.icd9 import Node as ICDNode
from preprocess import get_roots
from preprocess import icd_nodes
from preprocess import mimic_to_norm_icd
from pyathena.connection import Connection as AthenaConn
from pyathena.util import as_pandas
from typing import Callable
from typing import Iterable
from typing import List
from typing import Dict
from utils import get_conn


def get_icd(conn_func: Callable[[], AthenaConn]) -> List[str]:
    with conn_func() as CONN:
        cursor = CONN.cursor()
        query = "select icd9_code from mimiciii.diagnoses_icd"
        icd_codes = as_pandas(cursor.execute(query))
    return icd_codes["icd9_code"].tolist()


def note_lengths(docs: Iterable[str]) -> Iterable[int]:
    """Determine the number of words in each doc."""
    return map(lambda s: len(re.findall(r'\w+', s)))


def summary_table(conn_func: Callable[[], AthenaConn]) -> pd.DataFrame:
    """Return a summary table of the data."""
    def count_query(group_field: str) -> str:
        """Define a query to count fields grouped by the given field."""
        return f"""
        SELECT
          {group_field} as Category,
          count(distinct p.subject_id) as Patients,
          count(distinct a.hadm_id) as Admissions,
          count(distinct d.icd9_code) as 'ICD9 Codes',
          count(distinct a.deathtime) as Deaths
        FROM
          mimiciii.patients AS p
        LEFT JOIN
          mimiciii.diagnoses_icd AS d
        ON
          p.subject_id = d.subject_id
        LEFT JOIN
          mimiciii.admissions as a
        ON
          p.subject_id = a.subject_id
        GROUP BY
            {group_field};
        """
    # define grouping queries
    gender_query = count_query("p.gender")
    insurance_query = count_query("a.insurance")

    # retrieve data
    with conn_func() as CONN:
        cursor = CONN.cursor()
        gender_df = as_pandas(cursor.execute(gender_query))
        insurance_df = as_pandas(cursor.execute(insurance_query))

    # rename gender variables
    gender_df["Category"] = gender_df["Category"].replace(to_replace="M",
                                                          value="Male")
    gender_df["Category"] = gender_df["Category"].replace(to_replace="F",
                                                          value="Female")

    # derive totals
    totals_df = pd.DataFrame(gender_df.drop("Category", axis=1)
                             .apply(sum)).transpose()
    totals_df["Category"] = ["Totals"]

    # combine data by appending columns
    summary_df = pd.concat([totals_df, gender_df, insurance_df], axis=0)

    # derive other features
    def x_per_y(lox: Iterable[int], loy: Iterable[int]) -> List[float]:
        """Return the element-wise ratios of the given iterables."""
        return list(map(lambda x, y: round(x/y, 2), lox, loy))

    summary_df["Admissions Per Patient"] = x_per_y(summary_df["Admissions"],
                                                   summary_df["Patients"])

    summary_df["ICD9 Codes Per Patient"] = x_per_y(summary_df["ICD9 Codes"],
                                                   summary_df["Patients"])
    summary_df["ICD9 Codes Per Admission"] = x_per_y(summary_df["ICD9 Codes"],
                                                     summary_df["Admissions"])
    summary_df["Death Per Patient"] = x_per_y(summary_df["Deaths"],
                                              summary_df["Patients"])

    # subset desired features and reformat df
    final_df = summary_df.sort_values(by="Patients", ascending=False)\
                         .set_index("Category").transpose()

    return final_df


def count_children(node: ICDNode) -> int:
    """Count the total number of descendents from this node."""
    if node.children:
        ft.reduce(lambda acc, n: acc + count_children(n), node.children, 1)
    else:
        return 1


def icd_summary(codes: List[str], tree: ICD9) -> pd.DataFrame:
    """Summarize top level icd codes."""
    # convert codes to ICD9 objects
    filt_codes = [str(c) for c in codes if c]
    norm_codes = mimic_to_norm_icd(filt_codes)
    nodes = icd_nodes(norm_codes, tree)
    root_codes = get_roots(nodes)

    # count by root code
    root_freqs = Counter(root_codes)
    roots = list(root_freqs.keys())
    counts = list(root_freqs.values())

    # get summary table data
    code_names = [c.code for c in roots]
    descs = [c.description.strip() for c in roots]
    n_children = [count_children(r) for r in roots]
    data = {"Code": code_names, "Description": descs,
            "Mimic-iii Counts": counts, "Nodes in ICD Tree": n_children}
    df = pd.DataFrame(data)
    return df


def main() -> Dict:
    """Cache function results."""
    # declare constants
    data_dir = os.path.join(PROJ_DIR, "data")

    # get all icd codes
    print("Getting all icd codes .....")
    icd_fp = os.path.join(data_dir, "icd.csv")
    if os.path.exists(icd_fp):
        icd_codes = pd.read_csv(icd_fp, squeeze=True).tolist()
    else:
        icd_codes = get_icd(get_conn)
        pd.Series(icd_codes).to_csv(icd_fp, header=False)

    # get the root for each icd code
    print("Getting roots .....")
    roots_fp = os.path.join(data_dir, "roots.csv")
    if os.path.exists(roots_fp):
        roots = pd.read_csv(roots_fp)
    else:
        roots = get_roots(icd_codes)

    # get icd summary table
    print("Generating icd summary table .....")
    icd_table_fp = os.path.join(data_dir, "icd_summary.csv")
    if os.path.exists(icd_table_fp):
        icd_table = pd.read_csv(icd_table_fp)
    else:
        icd_table = icd_summary(icd_codes, TREE)
        icd_table.to_csv(icd_table_fp,
                         header=True, index=False)

    # get full summary table
    print("Generating full summary table .....")
    summary_fp = os.path.join(data_dir, "full_summary.csv")
    if os.path.exists(summary_fp):
        full_summary = pd.read_csv(summary_fp, index_col="Category")
    else:
        full_summary = summary_table(get_conn)
        full_summary.to_csv(summary_fp,
                            header=True, index=True)

    out_dict = {"icd_codes": icd_codes,
                "icd_table": icd_table,
                "full_summary": full_summary}
    return out_dict


if __name__ == "__main__":
    main()
