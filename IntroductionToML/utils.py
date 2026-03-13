from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import is_regressor
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import _is_arraylike, _num_features


def scatter_plot_with_feature_annotation(
    dataset: pd.DataFrame,
    feature_1: str,
    feature_2: str,
    target: pd.Series,
    sample_point: int = 60,
):
    # visualize the dataset
    plt.figure(figsize=(10, 6))

    # plot each data point (x = feature1, y = feature2) and the species in color
    ax = sns.scatterplot(
        data=dataset,
        x=feature_1,
        y=feature_2,
        hue=target,
        style=target,
        s=50,
    )

    # select a single data point
    sample = dataset.iloc[sample_point]

    # add text to point to single data point
    ax.annotate(
        f"({feature_1}, {feature_2})",
        (sample[feature_1] + 0.1, sample[feature_2] - 0.05),
        xytext=(sample[feature_2] + 0.5, sample[feature_2] - 0.3),
        fontsize=12,
        arrowprops={
            "width": 1,
            "headwidth": 6,
            "headlength": 6,
            "edgecolor": "black",
            "facecolor": "black",
        },
    )

    return ax


def scatter_plot_with_test_point(
    dataset: pd.DataFrame,
    feature_1: str,
    feature_2: str,
    target: pd.Series,
    test_point: tuple | list,
):
    plt.figure(figsize=(10, 6))

    # plot the species type in color
    ax = sns.scatterplot(
        data=dataset,
        x=feature_1,
        y=feature_2,
        hue=target,
        style=target,
        s=50,
    )

    # plot the new test point
    ax.scatter(x=[test_point[0]], y=[test_point[1]], color="deepskyblue")

    # add "new" text and arrow pointing at new test point
    ax.annotate(
        "New",
        (test_point[0] - 0.05, test_point[1] + 0.02),
        xytext=(test_point[0] - 1, test_point[1] + 0.3),
        fontsize=12,
        color="deepskyblue",
        arrowprops={
            "width": 1,
            "headwidth": 6,
            "headlength": 6,
            "edgecolor": "deepskyblue",
            "facecolor": "deepskyblue",
        },
    )

    return ax


def scatter_plot_with_lines_to_test_point(
    dataset: pd.DataFrame,
    feature_1: str,
    feature_2: str,
    target: pd.Series,
    test_point: tuple | list,
):
    plt.figure(figsize=(10, 6))

    # plot the species type in color
    ax = sns.scatterplot(
        data=dataset,
        x=feature_1,
        y=feature_2,
        hue=target,
        style=target,
        s=50,
        zorder=2,
    )

    # plot a line from the test point to each point in the dataset
    for _, flower in dataset.iterrows():
        ax.plot(
            [test_point[0], flower[feature_1]],
            [test_point[1], flower[feature_2]],
            color="gray",
            linewidth=1,
            alpha=0.5,
            zorder=1,
        )

    # plot the test point
    ax.scatter(
        x=[test_point[0]],
        y=[test_point[1]],
        color="deepskyblue",
        s=100,
        zorder=2,
    )

    return ax


def compute_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def get_closest_point(
    dataset: pd.DataFrame,
    feature_1: str,
    feature_2: str,
    test_point: tuple | list,
):
    distances = dataset.apply(
        lambda row: compute_distance(
            (row[feature_1], row[feature_2]),
            test_point,
        ),
        axis=1,
    )

    closest_index = distances.idxmin()
    return dataset.loc[closest_index]


def scatter_plot_with_closest_point_to_test_point(
    dataset: pd.DataFrame,
    feature_1: str,
    feature_2: str,
    target: pd.Series,
    test_point: tuple | list,
):
    plt.figure(figsize=(10, 6))

    # plot the dataset
    ax = sns.scatterplot(
        data=dataset,
        x=feature_1,
        y=feature_2,
        hue=target,
    )

    # plot the test point as an 'x'
    sns.scatterplot(
        x=[test_point[0]],
        y=[test_point[1]],
        color="deepskyblue",
        s=100,
        label="test point",
        ax=ax,
    )

    closest_point = get_closest_point(
        dataset,
        feature_1,
        feature_2,
        test_point,
    )

    ax.plot(
        [test_point[0], closest_point[feature_1]],
        [test_point[1], closest_point[feature_2]],
        color="gray",
        linewidth=1,
        alpha=0.5,
        zorder=0,
    )

    # plot a ring around the nearest datapoint
    sns.scatterplot(
        x=[closest_point[feature_1]],
        y=[closest_point[feature_2]],
        marker="o",
        label="nearest training point",
        edgecolor="black",
        facecolor="none",
        ax=ax,
        zorder=1,
    )

    return ax


def _is_arraylike_not_scalar(array):
    return _is_arraylike(array) and not np.isscalar(array)


def _check_boundary_response_method(estimator, response_method):
    has_classes = hasattr(estimator, "classes_")
    if has_classes and _is_arraylike_not_scalar(estimator.classes_[0]):
        msg = "Multi-label and multi-output multi-class classifiers are not supported"
        raise ValueError(msg)

    if has_classes and len(estimator.classes_) > 2:
        if response_method not in {"auto", "predict"}:
            msg = (
                "Multiclass classifiers are only supported when response_method is"
                " 'predict' or 'auto'"
            )
            raise ValueError(msg)
        methods_list = ["predict"]
    elif response_method == "auto":
        methods_list = ["decision_function", "predict_proba", "predict"]
    else:
        methods_list = [response_method]

    prediction_method = [getattr(estimator, method, None) for method in methods_list]
    prediction_method = reduce(lambda x, y: x or y, prediction_method)
    if prediction_method is None:
        raise ValueError(
            f"{estimator.__class__.__name__} has none of the following attributes: "
            f"{', '.join(methods_list)}."
        )

    return prediction_method


def _plot_decision_boundary(
    estimator,
    X,
    *,
    grid_resolution=100,
    eps=1.0,
    plot_method="contourf",
    response_method="auto",
    xlabel=None,
    ylabel=None,
    ax=None,
    **kwargs,
):
    if not grid_resolution > 1:
        raise ValueError(
            f"grid_resolution must be greater than 1. Got {grid_resolution} instead."
        )

    if not eps >= 0:
        raise ValueError(f"eps must be greater than or equal to 0. Got {eps} instead.")

    possible_plot_methods = ("contourf", "contour", "pcolormesh")
    if plot_method not in possible_plot_methods:
        available_methods = ", ".join(possible_plot_methods)
        raise ValueError(
            f"plot_method must be one of {available_methods}. "
            f"Got {plot_method} instead."
        )

    num_features = _num_features(X)
    if num_features != 2:
        raise ValueError(f"n_features must be equal to 2. Got {num_features} instead.")

    x0, x1 = _safe_indexing(X, 0, axis=1), _safe_indexing(X, 1, axis=1)

    x0_min, x0_max = x0.min() - eps, x0.max() + eps
    x1_min, x1_max = x1.min() - eps, x1.max() + eps

    xx0, xx1 = np.meshgrid(
        np.linspace(x0_min, x0_max, grid_resolution),
        np.linspace(x1_min, x1_max, grid_resolution),
    )

    if hasattr(X, "iloc"):
        # we need to preserve the feature names and therefore get an empty dataframe
        X_grid = X.iloc[[], :].copy()
        X_grid.iloc[:, 0] = xx0.ravel()
        X_grid.iloc[:, 1] = xx1.ravel()
    else:
        X_grid = np.c_[xx0.ravel(), xx1.ravel()]

    pred_func = _check_boundary_response_method(estimator, response_method)
    response = pred_func(X_grid)

    # convert classes predictions into integers
    if pred_func.__name__ == "predict" and hasattr(estimator, "classes_"):
        encoder = LabelEncoder()
        encoder.classes_ = estimator.classes_
        response = encoder.transform(response)

    if response.ndim != 1:
        if is_regressor(estimator):
            raise ValueError("Multi-output regressors are not supported")

        # TODO: Support pos_label
        response = response[:, 1]

    if xlabel is None:
        xlabel = X.columns[0] if hasattr(X, "columns") else ""

    if ylabel is None:
        ylabel = X.columns[1] if hasattr(X, "columns") else ""

    if plot_method not in ("contourf", "contour", "pcolormesh"):
        raise ValueError("plot_method must be 'contourf', 'contour', or 'pcolormesh'")

    if ax is None:
        _, ax = plt.subplots()

    plot_func = getattr(ax, plot_method)

    plot_func(xx0, xx1, response.reshape(xx0.shape), **kwargs)

    if xlabel is not None or not ax.get_xlabel():
        xlabel = xlabel if xlabel is None else xlabel
        ax.set_xlabel(xlabel)

    if ylabel is not None or not ax.get_ylabel():
        ylabel = ylabel if ylabel is None else ylabel
        ax.set_ylabel(ylabel)

    return ax


def plot_decision_boundary(
    estimator,
    X,
    *,
    grid_resolution=100,
    eps=1.0,
    plot_method="contourf",
    response_method="auto",
    xlabel=None,
    ylabel=None,
    ax=None,
    **kwargs,
):
    # plot the decision regions of the linear classifier
    ax = _plot_decision_boundary(
        estimator,
        X,
        ax=ax,
        grid_resolution=grid_resolution,
        eps=eps,
        plot_method=plot_method,
        response_method=response_method,
        xlabel=xlabel,
        ylabel=ylabel,
        **kwargs,
    )
    #
    # # plot the decision boundary of the linear classifier
    # _plot_decision_boundary(
    #     estimator,
    #     X,
    #     plot_method="contour",
    #     levels=[0, 1],
    #     colors="black",
    #     ax=ax,
    # )

    return ax
