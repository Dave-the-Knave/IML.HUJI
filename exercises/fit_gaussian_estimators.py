from gaussian_estimators import *
import plotly.graph_objects as go

if __name__ == '__main__':

    # Question 3.1-1
    sample = np.random.normal(10, 1, [1000, ])
    model = UnivariateGaussian(False).fit(sample)
    print(model.mu_, model.var_)

    # Question 3.1-2
    models = np.ndarray([100, ])
    for i in range(1, 100):
        models[i] = UnivariateGaussian(False).fit(sample[1:i*10, ]).mu_
    go.Figure() \
        .add_traces([go.Scatter(x=np.linspace(10, 1000, 100), y=np.abs(models - 10), mode='lines', marker=dict(color="black"), showlegend=False)]) \
        .update_layout(title_text=r"$\text{Expectation Error Given Sample Size}$", height=400, width=600) \
        .update_xaxes(title_text="Sample Size") \
        .update_yaxes(title_text="Expectation Error") \
        .show()

    # Question 3.1-3
    pdf = model.pdf(sample)
    go.Figure() \
        .add_traces([go.Scatter(x=sample, y=pdf, mode='markers',
                                marker=dict(color="black"), showlegend=False)]) \
        .update_layout(title_text=r"$\text{Estimated Function}$", height=400, width=600) \
        .update_xaxes(title_text="Input") \
        .update_yaxes(title_text="Probability Density") \
        .show()


    # Question 3.2-4
    mean = np.array([0, 0, 4, 0]).T
    covariance = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    sample = np.random.multivariate_normal(mean, covariance, 1000)
    model = MultivariateGaussian().fit(sample)
    print("estimated expectation:")
    print(model.mu_)
    print("estimated covariance matrix:")
    print(model.cov_)

    # Question 3.2-5
    heatmap = np.ndarray(shape=[200, 200])
    f1, f3 = np.linspace(-10, 10, 200), np.linspace(-10, 10, 200)
    for i in range(0, 200):
        for j in range(0, 200):
            heatmap[i, j] = model.log_likelihood(np.array([f1[i], 0, f3[j], 0]), covariance, sample)
    go.Figure(go.Heatmap(x=f1, y=f3, z=heatmap), layout=go.Layout(title="Likelihood Heatmap", height=400, width=400)) \
        .update_xaxes(title_text="f3 parameter") \
        .update_yaxes(title_text="f1 parameter") \
        .show()

    # Question 3.2-6
    best = np.max(heatmap)
    print("Highest log likelihood:{}".format(best))
    params = np.unravel_index(heatmap.argmax(), heatmap.shape)
    print("With parameters:")
    print(f1[params[0]], f3[params[1]])
