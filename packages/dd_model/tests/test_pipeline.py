from __future__ import absolute_import
import numpy as np
from dd_model.processing import data_management as dm
from dd_model.pipeline import obs_pipe

from duplicate_detection_model.config.core import config


def test_pipeline():
    # Given
    df_test = dm.load_test_dataset(file_name = config.app_config.test_data_file, nrows = 25)
    X_transformed = obs_pipe[:3].transform(df_test[['obs_id', 'short_desc', 'sub_sys', 'create_date']])

    # Then
    assert X_transformed.shape[0] == 25
    assert X_transformed.shape[1] == 4