"""Execute project scripts via command line arguments."""
import os
import sys

import pandas as pd

from scripts.models import train_models
from scripts.preprocess import (group_data, retrieve_icd, retrieve_notes)
from gensim.models import KeyedVectors
from utils import PROJ_DIR, TREE, get_conn


def main() -> None:
    """Cache results from time consuming processes."""

    # initialize directories
    datadir = os.path.join(PROJ_DIR, "data", "preprocessed")
    modeldir = os.path.join(PROJ_DIR, "data", "models")
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    # define filepaths
    roots_fp = os.path.join(datadir, "roots.pd")
    notes_fp = os.path.join(datadir, "notes.pd")
    model_fp = os.path.join(datadir, "model.pd")
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
        roots_df = pd.read_pickle(roots_fp)
        notes_df = pd.read_pickle(notes_fp)
        model_df = group_data(roots_df, notes_df)
        model_df.to_pickle(model_fp)

    if "--baseline" in sys.argv or "--lstm" in sys.argv:
        # get model data
        print("Loading prepped data .....")
        model_df = pd.read_pickle(model_fp).sample(100)

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
