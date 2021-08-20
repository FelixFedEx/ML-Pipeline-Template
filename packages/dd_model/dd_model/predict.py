from __future__ import absolute_import

import logging
import os
import typing as t

import pandas as pd

from dd_model import __version__ as _version
from dd_model import pipeline
from dd_model.config.core import ARTIFACTORY_DIR, config
from dd_model.processing import data_management as dm
from dd_model.processing.validation import validate_inputs

_logger = logging.getLogger(__name__)

_obs_pipe = pipeline.obs_pipe

file_name = os.environ.get("EXPLORE_DATA_FILE", config.app_config.explore_data_file)
df_explore = dm.load_explore_dataset(file_name=file_name)

_obs_pipe["simple_score_calculator"].df_exp = df_explore
_obs_pipe["simple_score_calculator"].search_period = config.model_config.prediction_search_period
_obs_pipe["simple_score_calculator"].is_open = config.model_config.prediction_status_is_open

_obs_pipe["score_predictor"].threshold = config.model_config.prediction_result_threshold
_obs_pipe["score_predictor"].n_dup = config.model_config.prediction_result_number


def make_prediction(*, input_data: t.Union[pd.DataFrame, dict],) -> dict:
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)
    results = {
        "predictions": None,
        "package_name": config.app_config.package_name,
        "version": _version,
        "errors": errors,
    }

    if not errors:
        predictions = _obs_pipe.transform(
            X=validated_data[config.model_config.features]
        )
        _logger.info(
            f"Making predictions with model version: {_version} "
            f"Predictions: {predictions}"
        )
        results = {
            "predictions": predictions.score.values,
            "package_name": config.app_config.package_name,
            "version": _version,
            "errors": errors,
        }

    return results
