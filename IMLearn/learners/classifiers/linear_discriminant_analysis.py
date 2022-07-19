from typing import NoReturn

from ... import metrics
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        m, d = X.shape[0], X.shape[1]
        # obtain classes and pi
        self.classes_, mk = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)
        self.pi_ = mk / m
        # define mu and cov dimensions
        self.mu_ = np.ndarray([n_classes, d])
        self.cov_ = np.ndarray([d, d])
        # obtain mu and cov
        for k in range(n_classes):
            class_samples = X[np.where(y == self.classes_[k])]
            self.mu_[k] = class_samples.mean(axis=0).T
            self.cov_ += np.sum((class_samples - self.mu_[k]) @ (class_samples - self.mu_[k]).T)
        # normalize covariance matrix with unbiased denominator and calculate inverse
        self.cov_ = self.cov_ / (m - n_classes)
        self._cov_inv = np.linalg.pinv(self.cov_)
        self.fitted_ = True
        return

    def _predict(self, X: np.ndarray) -> np.ndarray:
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
        return np.array([self.classes_[np.argmin(self.likelihood(X)[:, i], axis=0)] for i in range(X.shape[0])])

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        # calculate likelihoods by class
        n_classes = self.mu_.shape[0]
        m = X.shape[0]
        likelihoods = np.ndarray([n_classes, m])
        for k in range(n_classes):
            ak = self._cov_inv @ self.mu_[k]
            bk = np.log(self.pi_[k]) - (self.mu_[k].T @ self._cov_inv @ self.mu_[k]) / 2
            likelihoods[k] = X.dot(ak.T) + bk
        return likelihoods

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
