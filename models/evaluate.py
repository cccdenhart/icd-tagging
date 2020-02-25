"""Determines evaluation metrics."""
import os
import pickle
from typing import List

import torch
import torch.nn.functional as F
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader

from models_b.constants import PROJ_DIR
from models_b.dataset import InfoScoreDataset
from models_b.mlp import Mlp
from models_b.preprocess import CONN


def get_perplexity(X_train: List[List[float]], X_test: List[List[float]], n_topics: int) -> float:
    """Trains a LDA model on the training set and evaluates perplexity on the test set."""
    lda = LatentDirichletAllocation(n_components=n_topics,
                                    max_iter=5, learning_method='online',
                                    learning_offset=50., random_state=0)
    lda.fit(X_train)
    perplexity: float = lda.perplexity(X_test)
    return perplexity


def main() -> None:
    """Evaluate model."""
    # build dataset
    print("Building dataset .....")
    dataset_fp = os.path.join(PROJ_DIR, "data", "dataset.pt")
    if not os.path.exists(dataset_fp):
        dataset = InfoScoreDataset(CONN, limit=1000)
        dataset.read_data()
        dataset.build_features()
        dataset.build_labels()
        ds_file = open(dataset_fp, 'wb')
        dataset.conn = None
        pickle.dump(dataset, ds_file)
        ds_file.close()
    else:
        ds_file = open(dataset_fp, 'rb')
        dataset = pickle.load(ds_file)
        ds_file.close()

    # split data into train/test
    print("Splitting into train/test .....")
    trainset, testset = dataset.train_test_split()

    # define fixed model params
    ACT_FNS = [F.relu] * 2 + [F.softmax]
    N_EPOCHS = 10
    BATCH_SIZE = 32
    LAYER_SIZES = (len(trainset[0][0].tolist()), 20, 10, 1)

    # train model
    print("Training model .....")
    trainloader = DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    """
    mlp = Mlp(LAYER_SIZES, ACT_FNS, N_EPOCHS)
    mlp.run_train(trainloader)
    """
    lm = LinearRegression().fit(trainset.features.tolist(), trainset.labels)

    # evaluate mlp
    print("Evaluating mlp .....")
    X_test, y_test = testset.features.tolist(), testset.labels
    y_pred = lm.predict(X_test)
    mlp = Mlp(LAYER_SIZES, ACT_FNS, N_EPOCHS)
    mlp.run_train(trainloader)

    # evaluate mlp
    print("Evaluating mlp .....")
    X_test, y_test = testset.features, testset.labels
    y_pred = mlp(X_test).tolist()
    mae = mean_absolute_error(y_test, y_pred)

    # write the trained model
    print("Saving trained model .....")
    """
    mlp_fp = os.path.join(PROJ_DIR, "data", "mlp.pt")
    torch.save(mlp.state_dict(), mlp_fp)
    """
    fp = os.path.join(PROJ_DIR, "data", "lm.sci")
    pickle.dump(lm, open(fp, 'wb'))
    mlp_fp = os.path.join(PROJ_DIR, "data", "mlp.pt")
    torch.save(mlp.state_dict(), mlp_fp)

    # write results to file
    print("Writing results to file .....")
    results_fp = os.path.join(PROJ_DIR, "models_b", "results.txt")
    result_file = open(results_fp, "w+")
    result_file.write("MAE: " + str(mae))

    print("Done!")


if __name__ == "__main__":
    main()
