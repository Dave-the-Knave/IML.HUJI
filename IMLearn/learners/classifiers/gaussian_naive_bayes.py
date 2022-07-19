from typing import NoReturn

from .. import MultivariateGaussian
from ... import metrics
from ...base import BaseEstimator
import numpy as np

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        m, d = X.shape[0], X.shape[1]
        # obtain pi
        self.classes_, mk = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)
        self.pi_ = mk / m
        self.mu_ = np.ndarray([n_classes, d])
        self.vars_ = np.ndarray([n_classes, d, d])
        # obtain mu and vars
        for k in range(n_classes):
            class_samples = X[np.where(y == self.classes_[k])]
            self.mu_[k] = class_samples.mean(axis=0)
            self.vars_[k] = (class_samples - self.mu_[k]).T @ (class_samples - self.mu_[k]) / (len(class_samples) - 1)
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
        return np.array([self.classes_[np.argmax(self.likelihood(X)[:, i], axis=0)] for i in range(X.shape[0])])

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
        n_classes = self.mu_.shape[0]
        m = X.shape[0]
        likelihoods = np.ndarray([n_classes, m])
        for k in range(n_classes):
            t = X[:, np.newaxis, :] - self.mu_[k]
            mahalanobis = np.sum(t.dot(np.linalg.pinv(self.vars_[k])) * t, axis=2).flatten()
            _, logdet = np.linalg.slogdet(self.vars_[k])
            likelihoods[k, ] = np.log(self.pi_[k]) - 0.5 * (logdet + mahalanobis)
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
