"""Execute project scripts via command line arguments."""
import os
import sys

import pandas as pd
import numpy as np
import torch

from scripts.models import train_baseline, train_lstm
from scripts.preprocess import (group_data, retrieve_icd, retrieve_notes, split_df)
from transformers import AutoTokenizer, AutoModel
from gensim.models import KeyedVectors
from utils import PROJ_DIR, TREE, get_conn, ICDDataset, Batcher


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
    class_fp = os.path.join(procdir, "class_names.csv")
    train_fp = os.path.join(procdir, "train.pd")
    test_fp = os.path.join(procdir, "test.pd")
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
        notes_df = pd.read_pickle(notes_fp).sample(50000)
        model_df, class_names = group_data(roots_df, notes_df, word2vec, tokenizer)
        pd.Series(class_names).to_csv(class_fp, header=False, index=False)
        model_df.to_pickle(model_fp)

    if "--sub" in sys.argv:
        # subset the final dataframe for simpler modeling
        num_rows = 50000
        sub_df = pd.read_pickle(model_fp).sample(num_rows)
        sub_df.to_pickle(sub_fp)

    if "--split" in sys.argv:
        # split the final dataframe into train/test
        sub_df = pd.read_pickle(sub_fp)
        train_df, test_df = split_df(sub_df)
        train_df.to_pickle(train_fp)
        test_df.to_pickle(test_fp)

    if "--baseline" in sys.argv or "--lstm" in sys.argv:
        # get model data
        print("Loading prepped data .....")
        train_df = pd.read_pickle(train_fp)
        Y = train_df["roots"].tolist()
        X_d2v = np.array([np.array(d) for d in train_df["d2v"].tolist()])
        X_w2v = train_df["w2v_idx"].tolist()
        X_bert = train_df["bert_idx"].tolist()

        if "--baseline" in sys.argv:
            print("Training model .....")
            clfs = train_baseline(X_d2v, Y)
        else:
            if "--w2v" in sys.argv:
                # load word2vec embeddings
                print("Loading embeddings .....")
                word2vec = KeyedVectors.load_word2vec_format(w2v_fp, binary=True)
                embeddings = torch.Tensor(word2vec.vectors)
                X = X_w2v
            elif "--bert" in sys.argv:
                # load bert embeddings
                print("Loading embeddings .....")
                embeddings = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
                embeddings.eval()
                X = X_bert
            else:
                raise ValueError("No embeddings for lstm specified.")

            # train lstm
            print("Training model .....")
            clfs = train_lstm(X, Y, embeddings)

        # save models
        print("Saving models .....")
        for clf in clfs:
            clf.save(modeldir)


if __name__ == "__main__":
    main()
