from __future__ import absolute_import

import logging

from sklearn.pipeline import Pipeline

from dd_model.processing import preprocessors as dp

_logger = logging.getLogger(__name__)


# Pipeline
obs_pipe = Pipeline(
    [
        (
            "extract_fail_symptom_from_short",
            dp.SklearnTransformerWrapper(
                variables=["short_desc"], transformer=dp.ExtractFailSymptomFromShort()
            ),
        ),
        (
            "word_tokenizer",
            dp.SklearnTransformerWrapper(
                variables=["short_desc"], transformer=dp.WordTokenizer()
            ),
        ),
        (
            "word_lemmatizer",
            dp.SklearnTransformerWrapper(
                variables=["short_desc"], transformer=dp.WordLemmatizer()
            ),
        ),
        (
            "simple_score_calculator", 
            dp.SimpleScoreCalculator()
        ),
        (
            "score_predictor",
            dp.SklearnTransformerWrapper(
                variables=["score"], transformer=dp.ScorePredictor()
            ),
        ),
    ]
)
