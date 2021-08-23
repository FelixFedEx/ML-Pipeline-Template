from __future__ import absolute_import

import re

import numpy as np
import pandas as pd
import nltk


class DataCleaner:
    def __init__(self):
        return

    def extract_fail_symptom_from_short(self, df, col):
        pass

    def _strF2H(self, s):
        pass

    def _reg_func1(self, x):
        pass

    def _reg_func2(self, x):
        pass

    def _reg_func3(self, x):
        pass

    def _summrize_func(self, x1, x2, x3):
        pass

    def word_tokenizer(self, input_text: str) -> str:
        return process_token(str(input_text).lower())

    def extract_sio_from_updates(self, updates):
        pass

    def choose_first_dup_in_updates(self, obs_id, obs_id_array):
            pass


class DataHandler:
    def __init__(self):
        return

    def truncate_subsys(self, df, threshold=30):
        pass


class DataPipeline:
    def __init__(self):
        return
