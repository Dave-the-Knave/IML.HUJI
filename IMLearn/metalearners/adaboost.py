import numpy as np

from .. import metrics
from ..base import BaseEstimator
from typing import Callable, NoReturn


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = [], np.zeros(iterations), np.empty(iterations)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        m = X.shape[0]
        error_t = np.empty(self.iterations_)
        self.D_ = np.empty([self.iterations_, m])
        self.D_[0, :].fill(1/m)
        for T in range(self.iterations_ - 1):
            self.models_.append(self.wl_().fit(X, y*self.D_[T, :]))
            pred = self.models_[T].predict(X)
            misclass = 1 - np.abs((pred + y)/2)
            error_t[T] = np.dot(self.D_[T, :], misclass)
            self.weights_[T] = 0.5 * np.log((1/error_t[T])-1)
            for j in range(self.D_.shape[1]):
                self.D_[T + 1, j] = self.D_[T, j] * np.exp(-y[j] * self.weights_[T] * pred[j])
            self.D_[T+1] = self.D_[T+1] / np.sum(self.D_[T+1])

    def _predict(self, X):
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        hs = np.zeros(X.shape[0])
        for T in range(self.iterations_):
            hs += self.weights_[T] * self.models_[T]._predict(X)
        return np.sign(hs)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return metrics.loss_functions.misclassification_error(y, self._predict(X))

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        hs = np.zeros(X.shape[0])
        for i in range(T):
            hs += self.weights_[i] * self.models_[i]._predict(X)
        return np.sign(hs)

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return metrics.loss_functions.misclassification_error(y, self.partial_predict(X, T))
