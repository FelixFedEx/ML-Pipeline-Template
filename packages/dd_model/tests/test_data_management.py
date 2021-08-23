from __future__ import absolute_import
from dd_model.processing import data_management as dm
from dd_model.config.core import config

import pandas as pd


def test_data_management():

    df_explore = dm.load_explore_dataset(file_name = config.app_config.explore_data_file, nrows = 100)
    df_test = dm.load_test_dataset(file_name = config.app_config.test_data_file, nrows = 100)
    
    assert df_test.iloc[88].values[1] == 'Test Options selection menu'
    
    assert df_explore.iloc[37].values[1] == ['comply', 'specification', 'hinge', 'life', 'test', 'spec', 'degrees', 'minimum', 'fail',]
    assert df_explore.iloc[37].values[4] == pd.Timestamp('2019-01-01 21:06:15.670000')
    