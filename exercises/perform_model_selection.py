from __future__ import annotations
import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree
    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate
    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X = np.random.rand(n_samples) * 3.2 - 1.2
    true = (X + 3)*(X + 2)*(X + 1)*(X - 1)*(X - 2)
    y = true + np.random.randn(n_samples)*(noise**0.5)
    X_train, y_train, X_test, y_test = split_train_test(pd.DataFrame(X), pd.Series(y), 2/3)
    X_train, y_train, X_test, y_test = X_train.values.squeeze(), y_train.values.squeeze(), X_test.values.squeeze(), y_test.values.squeeze()
    fig = go.Figure()
    fig.add_traces(
        [go.Scatter(x=X_train, y=y_train, mode="markers", marker=dict(color="Blue"),
                    legendgroup="train set", name="train set", showlegend=True),
         go.Scatter(x=X_test, y=y_test, mode="markers", marker=dict(color="Red"),
                    legendgroup="test set", name="test set", showlegend=True),
         go.Scatter(x=X, y=true, mode="markers", marker=dict(color="Black"),
                    legendgroup="true model", name="true model", showlegend=True)])
    fig.update_xaxes(title_text="samples")
    fig.update_yaxes(title_text="labels")
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    t = [0 for x in range(11)]
    v = [0 for x in range(11)]
    for i in range(11):
        t[i], v[i] = cross_validate(PolynomialFitting(i), X_train, y_train, mean_square_error, 5)
    fig = go.Figure(data=[
        go.Bar(name='Train Error', x=[x for x in range(11)], y=t),
        go.Bar(name='Validation Error', x=[x for x in range(11)], y=v)
    ])
    fig.update_layout(barmode='group')
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k = v.index(min(v))
    print("the optimal degree is: ", k)
    model = PolynomialFitting(k).fit(X_train, y_train)
    print("it has test error of: ", mean_square_error(y_test, model.predict(X_test)))
    return


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions
    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate
    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    features, labels = sklearn.datasets.load_diabetes(return_X_y=True, as_frame=True)
    X_train, y_train, X_test, y_test = split_train_test(features, labels, n_samples /442)
    X_train, y_train, X_test, y_test = X_train.values, y_train.values, X_test.values, y_test.values

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    parameter = [x/200 - 1 for x in range(n_evaluations+1)]
    tr = [0 for x in range(n_evaluations)]
    vr = [0 for x in range(n_evaluations)]
    tl = [0 for x in range(n_evaluations)]
    vl = [0 for x in range(n_evaluations)]
    for i in range(n_evaluations):
        tr[i], vr[i] = cross_validate(RidgeRegression(lam=parameter[i]), X_train, y_train, mean_square_error, 5)
        tl[i], vl[i] = cross_validate(sklearn.linear_model.Lasso(alpha=parameter[i]), X_train, y_train, mean_square_error, 5)
    fig = go.Figure()
    fig.add_traces([go.Scatter(x=parameter, y=tr, mode="lines", marker=dict(color="Blue"),
                    legendgroup="ridge-train", name="ridge-train", showlegend=True),
         go.Scatter(x=parameter, y=vr, mode="lines", marker=dict(color="Green"),
                    legendgroup="ridge-validation", name="ridge-validation", showlegend=True),
         go.Scatter(x=parameter, y=tl, mode="lines", marker=dict(color="Red"),
                    legendgroup="lasso-train", name="lasso-train", showlegend=True),
         go.Scatter(x=parameter, y=vl, mode="lines", marker=dict(color="Orange"),
                    legendgroup="lasso-validation", name="lasso-validation", showlegend=True)])
    fig.update_xaxes(title_text="regularization parameter")
    fig.update_yaxes(title_text="errors")
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    ridge_test = mean_square_error(y_test, RidgeRegression(lam=0.065).fit(X_train, y_train).predict(X_test))
    lasso_test = mean_square_error(y_test, sklearn.linear_model.Lasso(alpha=0.48).fit(X_train, y_train).predict(X_test))
    linear_test = mean_square_error(y_test, LinearRegression().fit(X_train, y_train).predict(X_test))
    print("error on ridge regression: ", ridge_test)
    print("error on lasso regression: ", lasso_test)
    print("error on linear regression: ", linear_test)
    return


if __name__ == '__main__':
    np.random.seed(0)
    #select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()