#!/usr/bin/env python
# coding: utf-8

# ## Packages

import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from warnings import filterwarnings
from pprint import pprint
import gc
import dill

filterwarnings("ignore")

from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    Ridge,
    BayesianRidge,
    ARDRegression,
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
import xgboost as xgb
from sklearn.metrics import (
    make_scorer,
    get_scorer,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from tune_sklearn import TuneSearchCV
from scipy.stats import uniform, randint

# Reference: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter


# ## Global Parameters

ROOT_DIRECTORY = "/home/kaan.aytekin/Thesis"
# Non-feature columns
non_feature_columns = [
    "simulation_run",
    "is_accident_simulation",
    "accident_location",
    "accident_start_time",
    "accident_duration",
    "accident_lane",
    "prev_detector_detector_number",
    "next_detector_detector_number",
    "detector_number",
    "timestamp",
]


# ## UDFs


def sample_from_array(array, freq):
    array_size = len(array)
    sample_size = int(np.ceil(array_size * freq))
    array_slicer = np.zeros(array_size)
    test_index = np.random.choice(range(0, array_size), size=sample_size, replace=False)
    array_slicer[test_index] = 1
    return array[array_slicer.astype(bool)]


def kfolds_from_array(array, k, seed=None):
    if seed:
        np.random.seed(seed)
    np.random.shuffle(array)
    array_folds = np.array_split(array, k)
    return array_folds


def simulation_based_k_folds_split(df, k=10, seed=None):
    if seed:
        np.random.seed(seed)
    unique_simulation_combinations = (
        df[["simulation_run", "is_accident_simulation", "accident_lane"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    unique_simulation_combinations = (
        df[["simulation_run", "is_accident_simulation", "accident_lane"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    test_simulation_runs = (
        unique_simulation_combinations.groupby(
            ["is_accident_simulation", "accident_lane"]
        )
        .simulation_run.unique()
        .apply(lambda x: kfolds_from_array(x, k=k))
    )
    test_simulation_runs = test_simulation_runs.reset_index()

    for fold_number in range(k):
        complete_test_index = []
        for row in test_simulation_runs.itertuples():
            current_test_index = (
                (df.is_accident_simulation == row.is_accident_simulation)
                & (df.accident_lane == row.accident_lane)
                & (df.simulation_run.isin(row.simulation_run[fold_number]))
            )
            if len(complete_test_index):
                complete_test_index = complete_test_index | current_test_index
            else:
                complete_test_index = current_test_index

        train_index = ~complete_test_index
        test_index = complete_test_index
        # df_train = df[~complete_test_index].reset_index(drop=True)
        # df_test = df[complete_test_index].reset_index(drop=True)
        yield train_index, test_index  # df_train, df_test


def simulation_based_train_test_split(df, test_size=0.2, seed=None):
    """
    Splits {df} into train and test datasets by their simulation-type with given {test_size}
    """
    if seed:
        np.random.seed(seed)
    unique_simulation_combinations = (
        df[["simulation_run", "is_accident_simulation", "accident_lane"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    test_simulation_runs = (
        unique_simulation_combinations.groupby(
            ["is_accident_simulation", "accident_lane"]
        )
        .simulation_run.unique()
        .apply(lambda x: sample_from_array(x, freq=test_size))
    )
    test_simulation_runs = test_simulation_runs.reset_index()

    complete_test_index = []
    for row in test_simulation_runs.itertuples():
        current_test_index = (
            (df.is_accident_simulation == row.is_accident_simulation)
            & (df.accident_lane == row.accident_lane)
            & (df.simulation_run.isin(row.simulation_run))
        )
        if len(complete_test_index):
            complete_test_index = complete_test_index | current_test_index
        else:
            complete_test_index = current_test_index
    train_index = ~complete_test_index
    test_index = complete_test_index
    # df_train = df[~complete_test_index].reset_index(drop=True)
    # df_test = df[complete_test_index].reset_index(drop=True)
    return train_index, test_index  # df_train, df_test


def custom_cross_validation(
    models_list,
    performance_metrics_list,
    df_train,
    test_size=0.2,
    repetition_count=5,
    k_folds=10,
    seed=None,
):
    if seed:
        np.random.seed(seed)
    results = []
    for repetition in repetition_count:
        df_train, df_validate = simulation_based_train_test_split(
            df=df_train, test_size=test_size
        )
        x_train = df_train[feature_columns]
        y_train = df_train["target"]
        x_validate = df_validate[feature_columns]
        y_validate = df_validate["target"]
        for x, y in simulation_based_k_folds_split:
            for model in models_list:
                model.fit(x_train, y_train)
                y_predicted = model.predict(x_validate)
                for performance_metric in performance_metrics_list:
                    performance_metric(y_validate, y_predicted)


def custom_cross_validation(
    pipe,
    param_grid,
    performance_metric,
    df,
    test_size=0.2,
    repetition_count=5,
    k_folds=10,
    seed=None,
    n_jobs=-1,
    n_iter=10,
):
    from sklearn.metrics import get_scorer

    if seed:
        np.random.seed(seed)
    results = []
    for repetition in range(repetition_count):
        repetition_results = {"repetition": repetition}
        train_index, validation_index = simulation_based_train_test_split(
            df=df, test_size=test_size
        )
        df_train, df_validation = df[train_index], df[validation_index]
        x_train = df_train[FEATURE_COLUMNS]
        y_train = df_train["target"]
        x_validate = df_validation[FEATURE_COLUMNS]
        y_validate = df_validation["target"]
        inner_cv = list(simulation_based_k_folds_split(df=df_train, k=k_folds))
        # bayesian_optimizer = TuneSearchCV(
        optimizer = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_grid,
            scoring=performance_metric,
            n_jobs=n_jobs,
            cv=inner_cv,
            n_iter=n_iter,
            verbose=2,
            return_train_score=True,
            # search_optimization="random",#"bayesian",
        )
        cv_results = optimizer.fit(x_train, y_train)
        # cv_results = optimizer.cv_results_
        repetition_results["cv_results"] = cv_results
        best_model = optimizer.best_estimator_
        validation_predictions = best_model.predict(x_validate)
        scorer_function = get_scorer(performance_metric)._score_func
        score = scorer_function(y_validate, validation_predictions)
        repetition_results["best_model_validation_score"] = score
        results.append(repetition_results)
    return results


# ## Data Loading

df_train = pd.read_csv(
    os.path.join(ROOT_DIRECTORY, "data/thesis_data/x_train_processed.csv")
)
df_test = pd.read_csv(
    os.path.join(ROOT_DIRECTORY, "data/thesis_data/x_test_processed.csv")
)


processed_feature_columns_path = os.path.join(
    ROOT_DIRECTORY, "data/thesis_data/processed_feature_columns.txt"
)
with open(processed_feature_columns_path, "r") as reader:
    FEATURE_COLUMNS = reader.read().split("\n")


# ## Cross Validation: Hyper Parameter Tuning & Model Selection

tuning_pipeline = Pipeline(
    [("scaler", MinMaxScaler()), ("classifier", RandomForestRegressor())]
)

param_grid = [
    {
        "classifier": [LinearRegression()],
        "classifier__n_jobs": [30],
    },
    {
        "classifier": [Ridge()],
        "classifier__alpha": uniform(0, 10),  # (0,100),
    },
    {
        "classifier": [Lasso()],
        "classifier__alpha": uniform(0, 10),  # (0,100),
    },
    {
        "classifier": [BayesianRidge()],
        "classifier__alpha_1": uniform(0, 1),  # (0,1),
        "classifier__alpha_2": uniform(0, 1),  # (0,1),
    },
]


results = custom_cross_validation(
    pipe=tuning_pipeline,
    param_grid=param_grid,
    performance_metric="neg_mean_squared_error",
    df=df_train,
    test_size=0.2,
    repetition_count=5,
    k_folds=5,
    seed=5,
    n_jobs=50,
    n_iter=50,
)

with open(os.path.join(ROOT_DIRECTORY, "data/results/cv_lm.pkl"), "wb") as writer:
    dill.dump(obj=results, file=writer, recurse=True)
