from math import atan2

from IMLearn import metrics
from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.templates.default = "simple_white"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(f'../datasets/{filename}')
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        P = Perceptron().fit(X, y)
        losses = P.training_loss_

        # Plot figure
        go.Figure() \
            .add_traces([go.Scatter(x=[iteration for iteration in range(len(losses))], y=losses, mode='lines',
                                    marker=dict(color="black"), showlegend=False)]) \
            .update_layout(title_text=f"Perceptron Loss Over {n} Dataset", height=400, width=600) \
            .update_xaxes(title_text="Iteration") \
            .update_yaxes(title_text="Misclassifications") \
            .show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * np.pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black", showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit models and predict over training set
        linear_discriminator = LDA().fit(X, y)
        linear_predict = linear_discriminator.predict(X)
        naive_gaussian = GaussianNaiveBayes().fit(X, y)
        naive_predict = naive_gaussian.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy

        fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Linear Discriminant Analysis, {f}, Accuracy:{metrics.accuracy(y, linear_predict)}",
                                                            f"Gaussian Naive Bayes, {f}, Accuracy:{metrics.accuracy(y, naive_predict)}"))
        fig.add_traces([go.Scatter(x=X[:,0], y=X[:,1], mode="markers", showlegend=False,
                        marker=dict(color=linear_predict, symbol=y, line=dict(color="black", width=1))),
                        go.Scatter(x=X[:,0], y=X[:,1], mode="markers", showlegend=False,
                        marker=dict(color=naive_predict, symbol=y, line=dict(color="black", width=1),))],
                       rows=[1, 1], cols=[1, 2])
        fig.add_trace(go.Scatter(x=[naive_gaussian.mu_[0, 0], naive_gaussian.mu_[1, 0], naive_gaussian.mu_[2, 0]],
                        y=[naive_gaussian.mu_[0, 1], naive_gaussian.mu_[1, 1], naive_gaussian.mu_[2, 1]],
                        mode="markers", showlegend=False, marker=dict(symbol="x", color="black")),
                        row=1, col=2)
        fig.add_trace(get_ellipse(naive_gaussian.mu_[0], naive_gaussian.vars_[0]), row=1, col=2)
        fig.add_trace(get_ellipse(naive_gaussian.mu_[1], naive_gaussian.vars_[1]), row=1, col=2)
        fig.add_trace(get_ellipse(naive_gaussian.mu_[2], naive_gaussian.vars_[2]), row=1, col=2)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    #run_perceptron()
    compare_gaussian_classifiers()
