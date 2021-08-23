from __future__ import absolute_import

import ast
import logging
import typing as t
from datetime import timedelta

import pandas as pd
import pyodbc

from dd_model import pipeline
from dd_model.config.core import ARTIFACTORY_DIR, DATASET_DIR
from dd_model.processing import data_management as dm

_logger = logging.getLogger(__name__)


def load_explore_dataset(*, file_name: str, nrows: int = None) -> pd.DataFrame:
    if nrows is None:
        df = pd.read_csv(f"{DATASET_DIR}/{file_name}")
    else:
        df = pd.read_csv(f"{DATASET_DIR}/{file_name}", nrows=nrows)

    df["id"] = df["id"].astype("str")
    df["short"] = df["short"].apply(ast.literal_eval)
    df.loc[:, "create_date"] = pd.to_datetime(
        df["create_date"], format="%Y-%m-%d %H:%M:%S"
    )

    return df


def load_test_dataset(*, file_name: str, nrows: int = None) -> pd.DataFrame:
    if nrows is None:
        df = pd.read_csv(f"{DATASET_DIR}/{file_name}")
    else:
        df = pd.read_csv(f"{DATASET_DIR}/{file_name}", nrows=nrows)

    df["obs"] = df["obs"].astype("str")
    df["id_new"] = df["id_new"].astype("str")
    # df['create_date'] = pd.to_datetime(df['create_date'], format='%Y-%m-%d %H:%M:%S')

    return df


def update_new_data(df: pd.DataFrame):
    obs_pipe = pipeline.obs_pipe
    df_out = obs_pipe[:3].transform(df)

    return df_out
