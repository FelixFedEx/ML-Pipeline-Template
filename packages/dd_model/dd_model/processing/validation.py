from __future__ import absolute_import

import typing as t

import numpy as np
import pandas as pd
from marshmallow import Schema, ValidationError, fields, validates

from dd_model.config.core import config


class ObsDataInputSchema(Schema):
    short_desc = fields.Str()
    sub_sys = fields.Str()
    # if config.prediction_status_is_open:
    #    status = fields.Str()
    create_date = fields.DateTime()  # ('%Y-%m-%d %H:%M:%S')


def fill_na_time_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    validated_data = input_data.copy()
    if input_data[config.model_config.time_features].isnull().any().any():
        validated_data[config.model_config.time_features] = validated_data[
            config.model_config.time_features
        ].fillna(value=str(pd.Timestamp.now()))
    # if config.prediction_status_is_open:
    # input_data[config.model_config.optional_features].isnull().any().any():
    # validated_data[config.model_config.optional_features] = \
    # validated_data[config.model_config.optional_features].fillna(value='Open'))

    return validated_data


def validate_inputs(
    *, input_data: pd.DataFrame
) -> t.Tuple[pd.DataFrame, t.Optional[dict]]:
    """Check model inputs for unprocessable values."""

    validated_data = fill_na_time_inputs(input_data=input_data)

    # set many=True to allow passing in a list
    schema = ObsDataInputSchema(many=True)
    errors = None

    try:
        # replace numpy nans so that Marshmallow can validate
        schema.load(validated_data.replace({np.nan: None}).to_dict(orient="records"))
        validated_data["create_date"] = pd.to_datetime(validated_data["create_date"])
    except ValidationError as exc:
        errors = exc.messages

    return validated_data, errors
