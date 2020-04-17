"""Functions that generate specific data for visualizations."""
import functools as ft
import os
import re
from collections import Counter
from typing import Callable, Dict, Iterable, List

import pandas as pd

from icd9.icd9 import ICD9
from icd9.icd9 import Node as ICDNode
from utils import query_aws, V_CODE


def note_lengths(docs: Iterable[str]) -> Iterable[int]:
    """Determine the number of words in each doc."""
    return map(lambda s: len(re.findall(r'\w+', s)))


def summary_table(query_func: Callable[[str], pd.DataFrame]) -> pd.DataFrame:
    """Return a summary table of the data."""
    def count_query(group_field: str,
                    query_func: Callable[[str], pd.DataFrame]) -> str:
        """Define a query to count fields grouped by the given field."""
        query = f"""
        SELECT
          {group_field} as Category,
          count(distinct p.subject_id) as Patients,
          count(distinct a.hadm_id) as Admissions,
          count(distinct d.icd9_code) as "ICD9 Codes",
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
        return query_func(query)

    # retrieve queries
    gender_df = count_query("p.gender", query_func)
    insurance_df = count_query("a.insurance", query_func)

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

    # subset desired features and reformat df
    final_df = summary_df.sort_values(by="Patients", ascending=False)\
                         .set_index("Category").transpose()

    return final_df


def icd_summary(all_roots: List[str], tree: ICD9) -> pd.DataFrame:
    """Summarize top level icd codes."""
    # get non-V root nodes
    roots = [r for r in tree.children if r.code[0] not in ["V", "E"]]
    v_roots = [r for r in tree.children if r.code[0] == "V"]

    # count raw root in mimic
    counts_map = dict(Counter(all_roots))

    # get child counts for each root node
    leaf_map = {r.code: len(r.leaves) for r in roots}
    v_leaves = sum([len(v.leaves) for v in v_roots])
    leaf_map[V_CODE] = v_leaves

    # get summary table data
    desc_map = {r.code: r.description for r in roots}
    desc_map[V_CODE] = " ".join(["SUPPLEMENTARY CLASSIFICATION OF FACTORS",
                                 "INFLUENCING HEALTH STATUS AND CONTACT",
                                 "WITH HEALTH SERVICES"])

    # combine data
    root_codes = [r.code for r in roots]
    root_codes.append(V_CODE)
    counts = [counts_map[r] if r in counts_map.keys() else 0
              for r in root_codes]
    leaves = [leaf_map[r] for r in root_codes]
    descs = [desc_map[r] for r in root_codes]

    data = {"Code": root_codes, "Mimic-iii Counts": counts,
            "Number of Leaves": leaves, "Description": descs}
    df = pd.DataFrame(data)
    return df
