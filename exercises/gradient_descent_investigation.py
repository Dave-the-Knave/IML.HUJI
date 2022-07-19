import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from typing import Tuple, List, Callable, Type
from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import loss_functions
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm
    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted
    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path
    title: str, default=""
        Setting details to add to plot title
    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range
    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range
    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown
    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration
    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recording the objective's value and parameters
        at each iteration of the algorithm
    values: List[np.ndarray]
        Recorded objective values
    weights: List[np.ndarray]
        Recorded parameters
    """
    objectives = []
    parameters = []
    def call(solver, weights, val):
        objectives.append(val)
        parameters.append(weights)
    return call, objectives, parameters


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    norm = 0
    for module in [L1, L2]:
        norm += 1
        for eta in etas:
            callback, vals, weights = get_gd_state_recorder_callback()
            gd = GradientDescent(learning_rate=FixedLR(eta), callback=callback)
            gd = gd.fit(f=module(weights=init), X=None, y=None)
            descent_path = np.stack(weights, axis=0)
            # plot descent path
            fig = plot_descent_path(module=module, descent_path=descent_path, title=f"L{norm} with Fixed LR {eta}")
            fig.show()
            # plot convergence rates
            go.Figure() \
                .add_traces([go.Scatter(x=[x for x in range(len(vals))], y=vals, mode='markers',
                                        marker=dict(color="black"), showlegend=False)]) \
                .update_layout(title_text=f"Convergence Rate for L{norm} with Fixed LR {eta}", height=400, width=600) \
                .update_xaxes(title_text="Iterations") \
                .update_yaxes(title_text="Loss") \
                .show()
            # print minimum losses
            print(f"for L{norm} with fixed LR {eta} the minimum loss is: {min(vals)}")
    return


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    convergence = {}
    for gamma in gammas:
        callback, vals, weights = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=ExponentialLR(eta, gamma), callback=callback)
        gd = gd.fit(f=L1(weights=init), X=None, y=None)
        convergence[f"{gamma}"] = vals
        # print minimal losses
        print(f"for L1 with exponential LR at decay rate {gamma} the minimum loss is: {min(vals)}")

    # Plot algorithm's convergence for the different values of gamma
    x_axis = [x for x in range(1000)]
    fig = go.Figure()
    fig.add_traces([go.Scatter(x=x_axis, y=convergence["0.9"], mode="lines", marker=dict(color="Blue"),
                               legendgroup="Gamma 0.9", name="Gamma 0.9", showlegend=True),
                    go.Scatter(x=x_axis, y=convergence["0.95"], mode="lines", marker=dict(color="Green"),
                               legendgroup="Gamma 0.95", name="Gamma 0.95", showlegend=True),
                    go.Scatter(x=x_axis, y=convergence["0.99"], mode="lines", marker=dict(color="Red"),
                               legendgroup="Gamma 0.99", name="Gamma 0.99", showlegend=True),
                    go.Scatter(x=x_axis, y=convergence["1"], mode="lines", marker=dict(color="Orange"),
                               legendgroup="Gamma 1", name="Gamma 1", showlegend=True)])
    fig.update_xaxes(title_text="Iterations")
    fig.update_yaxes(title_text="Losses")
    fig.update_layout(title_text="L1 Convergence Rate with Different Decay Rates", height=600, width=800)
    fig.show()

    # Plot descent path for gamma=0.95
    callback, vals, weights = get_gd_state_recorder_callback()
    gd = GradientDescent(learning_rate=ExponentialLR(eta, 0.95), callback=callback)
    gd = gd.fit(f=L1(weights=init), X=None, y=None)
    descent_path = np.stack(weights, axis=0)
    fig = plot_descent_path(module=L1, descent_path=descent_path, title=f"L1 with Exponential LR 0.95")
    fig.show()
    return

def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion
    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset
    train_portion: float, default=0.8
        Portion of dataset to use as a training set
    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set
    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples
    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set
    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values

    # Plotting convergence rate of logistic regression over SA heart disease data
    callback, vals, weights = get_gd_state_recorder_callback()
    model = LogisticRegression(solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4), callback=callback))
    model.fit(X_train, y_train)
    #print("loss decrease: ", vals)
    
    fpr, tpr, thresholds = roc_curve(y_train, model.predict_proba(X_train))
    highest = 0
    for i in range(len(tpr)):
        #print(f"tpr {tpr[i]} fpr {fpr[i]} alpha {thresholds[i]}")
        if tpr[i] - fpr[i] > highest:
            alpha = thresholds[i]
            highest = tpr[i] - fpr[i]
            index = i
    print(f"best threshold is {alpha} at tpr {tpr[index]} and fpr {fpr[index]}")
    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         marker_color="rgb(49,54,149)",
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()
    print(f"at alpha={model.alpha_} loss is: ", model.loss(X_test, y_test))

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lam = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    train_scores_l1 = [0, 0, 0, 0, 0, 0, 0]
    validation_scores_l1 = [0, 0, 0, 0, 0, 0, 0]
    train_scores_l2 = [0, 0, 0, 0, 0, 0, 0]
    validation_scores_l2 = [0, 0, 0, 0, 0, 0, 0]
    best1 = np.inf
    best2 = np.inf
    for i in range(7):
        model1 = LogisticRegression(solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4)), penalty="l1", lam=lam[i])
        _, validation_scores_l1[i] = cross_validate(model1, X_train, y_train, loss_functions.misclassification_error, 5)
        print(f"for l1 regularization, lambda={lam[i]} has a crossvalidated missclassification of: {validation_scores_l1[i]}")
        if validation_scores_l1[i] < best1:
            best1 = validation_scores_l1[i]
            best_lam1 = lam[i]
        model2 = LogisticRegression(solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4)), penalty="l2", lam=lam[i])
        _, validation_scores_l2[i] = cross_validate(model2, X_train, y_train, loss_functions.misclassification_error, 5)
        print(f"for l2 regularization, lambda={lam[i]} has a crossvalidated missclassification of: {validation_scores_l2[i]}")
        if validation_scores_l2[i] < best2:
            best2 = validation_scores_l2[i]
            best_lam2 = lam[i]
    model1 = LogisticRegression(solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4)), penalty="l1", lam=best_lam1)
    model1.fit(X_train, y_train)
    error1 = model1.loss(X_test, y_test)
    model2 = LogisticRegression(solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4)), penalty="l2", lam=best_lam2)
    model2.fit(X_train, y_train)
    error2 = model2.loss(X_test, y_test)
    print(f"with l1 regularization, best lambda is {best_lam1} and test error is {error1}")
    print(f"with l2 regularization, best lambda is {best_lam2} and test error is {error2}")

if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
