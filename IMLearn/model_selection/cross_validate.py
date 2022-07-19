from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable, Type
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    # create list of indexes corresponding to indexes of X/y, labeled with integers in range cv
    # iterate through cv
    # load an array with indexes except for those labeled cv and a corresponding series
    # fit the estimator, predict and store the score
    # end iteration and average the scores
    m = X.shape[0]
    indices = np.arange(0, m)
    t_score = []
    v_score = []
    for i in range(cv):
        validation_indices = indices % cv == i
        k_minus_one_indices = indices % cv != i
        train = X[k_minus_one_indices]
        train_labels = y[k_minus_one_indices]
        validation = X[validation_indices]
        validation_labels = y[validation_indices]
        estimator.fit(train, train_labels)
        t_score.append(scoring(estimator.predict(train), train_labels))
        v_score.append(scoring(estimator.predict(validation), validation_labels))
    return sum(t_score)/len(t_score), sum(v_score)/len(v_score)
