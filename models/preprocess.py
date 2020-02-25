"""Functions for preprocessing clinical notes."""
import os
import pickle
import sys
from math import sqrt
from typing import Callable, List, Set, Tuple, Union

import numpy as np
import pandas as pd
from Levenshtein import distance
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pandas import DataFrame
from pyathena import connect
from pyathena.connection import Connection
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch import Tensor, tensor

from .constants import CONN, PROJ_DIR

# represents a document either as a string or numerical vector
Doc = Union[List[float], str]


def cust_cossim(text1: List[float], text2: List[float]) -> float:
    """Alters the signature of the sklearn cosine similarity function."""
    cs: float = cosine_similarity([text1], [text2])[0][0]
    return cs


def cust_led(text1: str, text2: str) -> float:
    """Alters the Levenshtein distance to return percent diff between strings."""
    led: int = distance(text1, text2)
    led_pct: float = led / max([len(text1), len(text2)])
    return led_pct


def get_categories(conn: Connection) -> List[str]:
    """Return distinct categories in the noteevents table."""
    query: str = "select distinct category from mimiciii.noteevents"
    raw_cats = conn.cursor().execute(query).fetchall()
    cats: List[str] = [n[0] for n in raw_cats]

    # to pickle
    cat_fp = os.path.join(PROJ_DIR, "data", "categories.list")
    cat_f = open(cat_fp, 'wb')
    pickle.dump(cats, cat_f)
    return cats


def cat_data(cat: str, conn: Connection, limit: int = 0) -> Tuple[List[int], List[str]]:
    """Get all or limited notes and patient ids in the given category, sorted by date."""
    limit_str: str = f" limit {limit};" if limit else ";"
    query: str = f"select subject_id, text from mimiciii.noteevents where category='{cat}' order by charttime{limit_str}"
    df: DataFrame = pd.read_sql(query, conn)
    patient_ids: List[int] = df["subject_id"].tolist()
    notes: List[str] = df["text"].tolist()
    return (patient_ids, notes)


def clean_notes(notes: List[str]) -> List[str]:
    """
    Clean the given notes with the below steps.

    - Remove stop words
    - Remove numerical strings
    - to lower case
    """
    stops: Set[str] = set(stopwords.words('english'))
    filt_toks: bool = lambda tok: (
        tok not in stops) and (not tok.isnumeric())
    new_notes = [" ".join(
        list(filter(filt_toks, word_tokenize(note)))).lower() for note in notes]
    return new_notes


def vectorize_notes(notes: List[str], cat: int) -> List[List[float]]:
    """Generate a tfidf matrix on the given notes."""
    tf_vectorizer = TfidfVectorizer(
        max_df=0.95, min_df=2, stop_words='english')
    tf: List[List[float]] = tf_vectorizer.fit_transform(
        notes).todense().tolist()

    # save tf model
    tf_fp = os.path.join(PROJ_DIR, "data", "tfs", f"tf_{cat}.sci")
    tf_f = open(tf_fp, "wb")
    pickle.dump(tf_vectorizer, tf_f)
    return tf


def get_topics(note_vect: List[List[float]], cat: int, n_topics: int = 5) -> List[List[float]]:
    """Extract topics from each note."""
    lda_model = LatentDirichletAllocation(n_components=n_topics,
                                          max_iter=5, learning_method='online',
                                          learning_offset=50., random_state=0)
    lda: List[List[float]] = lda_model.fit_transform(note_vect).tolist()

    # save lda model
    lda_fp = os.path.join(PROJ_DIR, "data", "ldas", f"lda_{cat}.sci")
    lda_f = open(lda_fp, "wb")
    pickle.dump(lda_model, lda_f)
    return lda


def note_similarity(notes: List[Doc],
                    patient_ids: List[int],
                    simf: Callable[[Doc, Doc], float]) -> List[float]:
    """
    Find similarity between sequential notes in each patient record.

    :param notes: a list of clinical notes as documents
    :param patient_ids: a list of patient ids corresponding to each note
    :param simf: a function measuring similarity between documents

    :returns all_sims: similarities between each doc less the first one in a record
    """
    all_sims: List[float] = []
    for patient_id in set(patient_ids):
        idxs = [idx for idx, i in enumerate(patient_ids) if i == patient_id]
        subset = [notes[idx] for idx in idxs]
        sims = [simf(subset[i], subset[i - 1])
                for i in range(1, len(subset))]
        all_sims += sims
    return all_sims


def max_tfidf(matrix: List[List[float]]) -> List[float]:
    """Get the max term of the documents for each term."""
    max_terms = [max(row) for row in np.transpose(matrix)]
    return max_terms


def tfidf_diff(notes: List[List[float]], patient_ids: List[int]) -> List[List[float]]:
    """
    Find the difference in term values between a note and its records prior max terms.

    A max tfidf for a patient record is found by maximizing each term from the tfidf matrix in all documents for a patient record.  The purpose of this is to identify for which terms a note is or isn't adding value.
    """
    all_diffs = []
    for patient_id in set(patient_ids):
        idxs = [idx for idx, i in enumerate(patient_ids) if i == patient_id]
        subset = [notes[idx] for idx in idxs]
        for i in range(1, len(subset)):
            note = subset[i]
            priors = subset[:i]
            max_terms = max_tfidf(priors)
            diff = [term - max_term for term, max_term in zip(note, max_terms)]
            all_diffs.append(diff)
    return all_diffs


def dim_reduce(vect_notes: List[List[float]], n_comps: int, cat: int) -> List[List[float]]:
    """Reduce the dimensionality of the given features via PCA."""
    pca: PCA = PCA(n_components=n_comps, svd_solver='auto')
    feats: List[List[float]] = pca.fit_transform(vect_notes).tolist()

    # save pca model
    pca_fp = os.path.join(PROJ_DIR, "data", "pcas", f"pca_{cat}.sci")
    pca_f = open(pca_fp, "wb")
    pickle.dump(pca, pca_f)
    return feats


def transform(notes: List[str], tf: TfidfVectorizer, pca: PCA) -> Tensor:
    """Transform a list of notes to be receivable by the mlp model."""
    clean_notes = clean_notes(notes)
    vect_notes = tf.transform(clean_notes)
    dim_notes = pca.transform(vect_notes)
    X = tensor(dim_notes)
    return X


def top_feats(topics: List[List[float]], vocab: List[str], n_words: int) -> List[List[str]]:
    """
    Return the top n words for each topic.

    :param topics: the components from the trained LDA object
    :param words: the feature names as returned by the Tfid object

    :returns top_words: the top words per topic
    """
    top_words = []
    for topic in topics:
        word_idx = np.argsort(topic)[:: -1][: n_words]
        top_words.append([vocab[i] for i in word_idx])
    return top_words


# constants for jupyter experimentation
DF = pd.read_sql("select * from mimiciii.noteevents limit 100", CONN)
"""
CATS = get_categories(CONN)
DATA = cat_data(CATS[5], CONN)
IDS = DATA[0]
NOTES = clean_notes(DATA[1])
VECTS = vectorize_notes(NOTES)
TOPICS = get_topics(VECTS)
"""
