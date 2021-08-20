from __future__ import absolute_import

import pandas as pd
import numpy as np

from dd_model.processing.validation import validate_inputs
from dd_model.processing import data_management as dm
from dd_model.config.core import config


def test_validate_inputs_scenario_1():
    # Given
    test_data = {
        "short_desc": ["asds abds kqjlk &&&"],
        "sub_sys": [123],
        "create_date": [None]
    }
    test_inputs = pd.DataFrame(data=test_data)

    # When
    validated_inputs, errors = validate_inputs(input_data=test_inputs)

    # Then
    assert errors != None
    assert errors[0]['sub_sys'][0] == 'Not a valid string.'


def test_validate_inputs_scenario_2():
    # Given
    test_data = {
        "short_desc": ["asds abds kqjlk &&&"],
        "sub_sys": ["Bios"],
        "create_date": ['2021-08-03 10:43']
    }
    test_inputs = pd.DataFrame(data=test_data)

    # When
    validated_inputs, errors = validate_inputs(input_data=test_inputs)

    # Then
    assert errors == None
    assert isinstance(validated_inputs.at[0, "create_date"], pd.Timestamp)
    

def test_validate_inputs_scenario_3():
    # Given
    test_data = {
        "short_desc": ["asds abds kqjlk &&&"],
        "sub_sys": [None],
        "create_date": [None]
    }
    test_inputs = pd.DataFrame(data=test_data)

    # When
    validated_inputs, errors = validate_inputs(input_data=test_inputs)

    # Then
    assert errors[0]['sub_sys'][0] == 'Field may not be null.'

    
def test_validate_inputs_scenario_4():
    # Given
    df_test = dm.load_test_dataset(file_name = config.app_config.test_data_file)

    # When
    validated_inputs, errors = validate_inputs(input_data=df_test[config.model_config.features])

    # Then
    assert errors == None