from __future__ import absolute_import

import re

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from dd_model.processing.utils.clean import lemmatize_verbs
from dd_model.processing.utils.data_handler import (
    DataCleaner,
    DataPipeline,
)

data_cleaner = DataCleaner()
data_pipeline = DataPipeline()


# Part1. Pre-processing
class SklearnTransformerWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper for Scikit-learn pre-processing transformers,
    like the SimpleImputer() or OrdinalEncoder(), to allow
    the use of the transformer on a selected group of variables.
    """

    def __init__(self, variables=None, transformer=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        self.transformer = transformer

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        The `fit` method allows scikit-learn transformers to
        learn the required parameters from the training data set.
        """

        self.transformer.fit(X[self.variables])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""
        X = X.copy()
        X[self.variables] = self.transformer.transform(X[self.variables])
        return X


class WordTokenizer(TransformerMixin, BaseEstimator):
    """
    :X: str
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        data_cleaner = DataCleaner()
        for col in X.columns:
            X[col] = X[col].apply(lambda x: data_cleaner.word_tokenizer(x))

        return X



class WordLemmatizer(TransformerMixin, BaseEstimator):
    """
    Lemmatize verbs only
    :X: list[str]
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col in X.columns:
            X[col] = X[col].apply(lambda x: lemmatize_verbs(x))

        return X


class WordJoiner(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].apply(lambda x: " ".join(x))

        return X
        