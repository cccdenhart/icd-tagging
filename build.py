"""Execute project scripts via command line arguments."""
import os
import sys

import pandas as pd

from scripts.models import train_models
from scripts.preprocess import (group_data, retrieve_icd, retrieve_notes, split_df)
from transformers import AutoTokenizer
from gensim.models import KeyedVectors
from utils import PROJ_DIR, TREE, get_conn


def main() -> None:
    """Cache results from time consuming processes."""

    # initialize directories
    datadir = os.path.join(PROJ_DIR, "data")
    procdir = os.path.join(datadir, "preprocessed")
    modeldir = os.path.join(datadir, "models")
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    # define filepaths
    roots_fp = os.path.join(procdir, "roots.pd")
    notes_fp = os.path.join(procdir, "notes.pd")
    model_fp = os.path.join(procdir, "model.pd")
    sub_fp = os.path.join(procdir, "sub.pd")
    w2v_fp = os.path.join(datadir, "embeddings",
                          "GoogleNews-vectors-negative300.bin")

    if "--roots" in sys.argv:
        # process icd codes
        roots_df = retrieve_icd(get_conn, TREE)
        roots_df.to_pickle(roots_fp)

    if "--notes" in sys.argv:
        # process notes
        notes_df = retrieve_notes(get_conn)
        notes_df.to_pickle(notes_fp)

    if "--prep" in sys.argv:
        # prepare the modeling df
        print("Loading embeddings .....")
        word2vec = KeyedVectors.load_word2vec_format(w2v_fp, binary=True)

        # load tokenizer
        print("Loading tokenizer .....")
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        # group data
        print("Grouping data .....")
        roots_df = pd.read_pickle(roots_fp)
        notes_df = pd.read_pickle(notes_fp)
        model_df = group_data(roots_df, notes_df, word2vec, tokenizer)
        model_df.to_pickle(model_fp)

    if "--sub" in sys.argv:
        # subset the final dataframe for simpler modeling
        num_rows = 50000
        sub_df = pd.read_pickle(model_fp).sample(num_rows)
        sub_df.to_pickle(sub_fp)

    if "--baseline" in sys.argv or "--lstm" in sys.argv:
        # get model data
        print("Loading prepped data .....")
        sub_df = pd.read_pickle(sub_fp)

        # load word2vec embeddings
        print("Loading embeddings .....")
        word2vec = KeyedVectors.load_word2vec_format(w2v_fp, binary=True)

        # get models, train, and save
        is_bl = "--baseline" in sys.argv
        print("Training model .....")
        clfs = train_models(model_df["roots"].tolist(),
                            model_df["tokens"].tolist(),
                            word2vec,
                            is_bl)

        # save models
        print("Saving models .....")
        for clf in clfs:
            clf.save(modeldir)

if __name__ == "__main__":
    main()
