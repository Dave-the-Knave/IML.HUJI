from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    df = df[df['price'] > 0]
    training_labels = df.price.copy(deep=True)
    df.drop(['price', 'id', 'lat', 'long'], axis=1, inplace=True)
    df = pd.concat([df, pd.get_dummies(df['zipcode'], prefix='zipcode_', dummy_na=False)],
                   axis=1).drop(['zipcode'], axis=1)
    df['date'] = pd.to_numeric(df['date'].str.split(pat='T').str[0])
    sale_year = df['date'].apply(lambda x: np.floor(x/10000))
    ren_bld = pd.concat([sale_year - df['yr_built'], sale_year - df['yr_renovated']], join='outer', axis=1)
    df['yr_renovated'] = ren_bld.min(axis=1)
    df['roomsize'] = df['sqft_living'] / (df['bedrooms'] + 1)
    df['bathroom_proportional'] = df['bedrooms'] / (df['bathrooms'] + 1)
    for k, v in df.iteritems():
        df[df[k].isnull() | df[k] == np.NaN] = 0
        minimum, maximum = v.min(), v.max()
        df[k] = (v - minimum)/(maximum - minimum)
    return df, training_labels


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for k, v in X.iteritems():
        pearson = np.cov(v, y)[0, 1]/(np.std(v)*np.std(y))
        go.Figure() \
            .add_traces([go.Scatter(x=v, y=y, mode='markers',
                                marker=dict(color="Blue"), showlegend=False)]) \
            .update_layout(title_text=f"Correlation between {k} and Price: {pearson}", height=500, width=500) \
            .update_xaxes(title_text=k) \
            .update_yaxes(title_text="Price") \
            .write_image(f"{output_path}/{k}_price_cor.jpg")
    return


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    features, labels = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    #feature_evaluation(features, labels, "../exercises/graphs")

    # Question 3 - Split samples into training- and testing sets.
    X_train, y_train, X_test, y_test = split_train_test(features, labels, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    X_train, y_train, X_test, y_test = X_train.to_numpy(na_value=0), y_train.to_numpy(na_value=0), X_test.to_numpy(na_value=0), y_test.to_numpy(na_value=0)
    avg_loss, std_loss = np.ndarray([90]), np.ndarray([90])
    p = [x for x in range(10, 100)]
    for i in p:
        loss = []
        for j in range(1, 10):
            sample = np.random.choice([x for x in range(1, X_train.shape[0])], size=int(i*X_train.shape[0]/100), replace=False)
            sample_X = X_train[sample, :]
            sample_y = y_train[sample]
            model = LinearRegression(include_intercept=True)
            model._fit(sample_X, sample_y)
            loss.append(model._loss(y_test, model._predict(X_test)))
        avg_loss[i-10] = np.mean(loss)
        std_loss[i-10] = np.std(loss)
    go.Figure() \
        .add_traces([go.Scatter(x=p, y=avg_loss, mode='markers', marker=dict(color="Blue"), showlegend=False),
            go.Scatter(x=p, y=avg_loss - 2 * std_loss, fill=None, mode="lines", line=dict(color="lightgrey"),
                       showlegend=False),
            go.Scatter(x=p, y=avg_loss + 2 * std_loss, fill='tonexty', mode="lines", line=dict(color="lightgrey"),
                       showlegend=False)]) \
        .update_layout(title_text=r"Relation btw Loss and Sample Size", height=500, width=500) \
        .update_xaxes(title_text=r"Sample Size (%)") \
        .update_yaxes(title_text=r"Mean Squared Error") \
        .show()
