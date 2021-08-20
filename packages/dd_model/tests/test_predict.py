from __future__ import absolute_import

import numpy as np
#from pandas import read_csv

from dd_model import __version__ as _version
from dd_model.config.core import config
from dd_model.processing import data_management as dm
from dd_model.predict import make_prediction
from dd_model.processing.evaluation import caculate_prediction_accuracy


def test_prediction_result():

    # Given
    df_test = dm.load_test_dataset(file_name = config.app_config.test_data_file)
    #df_test_label = read_csv(config.app_config.test_label_data_file)
    
    # When
    subject = make_prediction(input_data=df_test[config.model_config.features])
    y_pred = subject['predictions']
    y_true = df_test.dup_id_new.values
    accuracy = caculate_prediction_accuracy(y_true, y_pred)[0]

    # Then
    expected = {"predictions": None,
                "package_name": config.app_config.package_name,
                "version": _version,
                "errors": None}
    expected_accuracy = 0.076

    assert accuracy >= expected_accuracy
    for key in subject.keys():
        assert key in expected.keys()
    assert subject['package_name'] == expected['package_name']
    assert subject['version'] == expected['version']
    assert len(subject['predictions']) == 3000