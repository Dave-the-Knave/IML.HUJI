import numpy as np
from typing import Tuple

from IMLearn.learners import classifiers
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    model = AdaBoost(wl=classifiers.DecisionStump, iterations=n_learners)
    model = model.fit(train_X, train_y)

    train_loss, test_loss = [], []
    for i in range(n_learners):
        train_loss.append(model.partial_loss(train_X, train_y, T=i))
        test_loss.append(model.partial_loss(test_X, test_y, T=i))
    fig = go.Figure()
    fig.add_traces([go.Scatter(x=[x for x in range(n_learners)], y=train_loss, mode="markers", marker=dict(color="Blue"),
                             legendgroup="train loss", name="train loss", showlegend=True),
                    go.Scatter(x=[x for x in range(n_learners)], y=test_loss, mode="markers", marker=dict(color="Red"),
                             legendgroup="test loss", name="test loss", showlegend=True)])
    fig.update_xaxes(title_text="iterations")
    fig.update_yaxes(title_text="loss")
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    model_names = ["5 iterations", "50 iterations", "100 iterations", "250 iterations"]
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{x}}}$" for x in model_names],
                    horizontal_spacing = 0.01, vertical_spacing=.03)
    for i, m in enumerate(T):
        fig.add_traces([decision_surface(model.partial_predict, lims[0], lims[1], showscale=False, T=m-2),
                        go.Scatter(x=test_X[:,0], y=test_X[:,1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=(i//2) + 1, cols=(i % 2) + 1)
        fig.update_layout(title=rf"$\textbf{{Decision Boundaries of Ensembles at Different Sizes}}$", margin=dict(t=100))\
        .update_xaxes(visible=False). update_yaxes(visible=False)
        fig.show()

    # Question 3: Decision surface of best performing ensemble
    fig = go.Figure()
    fig.add_traces([decision_surface(model.partial_predict, lims[0], lims[1], showscale=False, T=199),
                        go.Scatter(x=test_X[:,0], y=test_X[:,1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))])
    fig.update_layout(title=rf"$\textbf{{Ensemble Size 200 with Accuracy 0.994}}$", margin=dict(t=100)) \
        .update_xaxes(visible=False). update_yaxes(visible=False)
    fig.show()

    # Question 4: Decision surface with weighted samples
    D = 10 * model.D_[n_learners-2, :] / np.max(model.D_[n_learners-2, :])
    fig = go.Figure()
    fig.add_traces([decision_surface(model.partial_predict, lims[0], lims[1], showscale=False, T=248),
                    go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=train_y, colorscale=[custom[0], custom[-1]], line=dict(color="black", width=1),
                               size=D))])
    fig.update_layout(title=rf"$\textbf{{Full Ensemble With Weighted Sample Sizes}}$", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0.4)
