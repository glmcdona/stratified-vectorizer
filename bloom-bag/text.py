
import array
from collections import defaultdict
from collections.abc import Mapping
from functools import partial
from numbers import Integral, Real
from operator import itemgetter
import re
import unicodedata
import warnings

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize
from sklearn.feature_extraction._hash import BloomBags
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import _VectorizerMixin, CountVectorizer
from sklearn.utils.validation import check_is_fitted, check_array, FLOAT_DTYPES
from sklearn.utils import _IS_32BIT, IS_PYPY
from sklearn.utils._param_validation import StrOptions, Interval
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class BloomVectorizer(
    TransformerMixin, _VectorizerMixin, BaseEstimator, auto_wrap_output_keys=None
):
    r"""TODO: Write docstring
    """

    _parameter_constraints: dict = {
        "input": [StrOptions({"filename", "file", "content"})],
        "encoding": [str],
        "decode_error": [StrOptions({"strict", "ignore", "replace"})],
        "strip_accents": [StrOptions({"ascii", "unicode"}), None, callable],
        "lowercase": ["boolean"],
        "preprocessor": [callable, None],
        "tokenizer": [callable, None],
        "stop_words": [StrOptions({"english"}), list, None],
        "token_pattern": [str, None],
        "ngram_range": [tuple],
        "analyzer": [StrOptions({"word", "char", "char_wb"}), callable],
        "n_features": [Interval(Integral, 1, np.iinfo(np.int32).max, closed="left")],
        "binary": ["boolean"],
        "norm": [StrOptions({"l1", "l2"}), None],
        "alternate_sign": ["boolean"],
        "dtype": "no_validation",  # delegate to numpy
    }

    def __init__(
        self,
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        analyzer="word",
        n_features=None,
        n_bags=5,
        error_rate=0.01,
        feature_rank=None,
        binary=False,
        norm="l2",
        dtype=np.float64,
    ):
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.n_features = n_features
        self.n_bags = n_bags
        self.error_rate = error_rate
        self.feature_rank = feature_rank
        self.ngram_range = ngram_range
        self.binary = binary
        self.norm = norm
        self.dtype = dtype
        self.bloom_bag = None

        if feature_rank is not None and n_features is not None:
            raise ValueError(
                "n_features and feature_rank cannot be set at the same time"
            )

    def partial_fit(self, X, y=None):
        """Only validates estimator's parameters.

        This method allows to: (i) validate the estimator's parameters and
        (ii) be consistent with the scikit-learn transformer API.

        Parameters
        ----------
        X : ndarray of shape [n_samples, n_features]
            Training data.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            HashingVectorizer instance.
        """
        # TODO: only validate during the first call
        self._validate_params()
        return self

    def _rank_features(self, X, y, n_features):
        """Run feature ranking.

        Parameters
        ----------
        X : ndarray of shape [n_samples, n_features]
            Training data.

        y : ndarray of shape [n_samples]
            Target values.

        n_features : int
            Number of features to select.

        Returns
        -------
        ranked_features : ndarray of shape [n_features]
            Ranked features.
        """
        if self.feature_rank is not None:
            return self.feature_rank

        pipeline = Pipeline(
            [
                (
                    "features",
                    CountVectorizer(tokenizer=self.tokenizer, max_features=n_features),
                ),
                ("classifier", LogisticRegression(solver="lbfgs", max_iter=1000)),
            ]
        )
        pipeline.fit(X, y)
        weights = pipeline.named_steps["classifier"].coef_
        feature_names = pipeline.named_steps["features"].vocabulary_
        feature_names = [
            feature
            for feature, index in sorted(feature_names.items(), key=lambda x: x[1])
        ]
        ordered_features = [
            feature
            for weight, feature in sorted(zip(weights[0], feature_names), reverse=True)
        ]

        return ordered_features

    def fit(self, X, y=None):
        """Only validates estimator's parameters.

        This method allows to: (i) validate the estimator's parameters and
        (ii) be consistent with the scikit-learn transformer API.

        Parameters
        ----------
        X : ndarray of shape [n_samples, n_features]
            Training data.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            HashingVectorizer instance.
        """
        self._validate_params()

        # triggers a parameter validation
        if isinstance(X, str):
            raise ValueError(
                "Iterable over raw text documents expected, string object received."
            )

        self._warn_for_unused_params()
        self._validate_ngram_range()

        # Run feature ranking
        self.ranked_features = self._rank_features(X, y, self.n_features)

        # Build the bloom bag
        self.bloom_bag = BloomBags(
            n_bags=self.n_bags,
            error_rate=self.error_rate,
            input_type="string",
            feature_rank=self.ranked_features,
        )

        analyzer = self.build_analyzer()
        self.bloom_bag.fit((analyzer(doc) for doc in X), y=y)
        return self

    def transform(self, X):
        """Transform a sequence of documents to a document-term matrix.

        Parameters
        ----------
        X : iterable over raw text documents, length = n_samples
            Samples. Each sample must be a text document (either bytes or
            unicode strings, file name or file object depending on the
            constructor argument) which will be tokenized and hashed.

        Returns
        -------
        X : sparse matrix of shape (n_samples, n_features)
            Document-term matrix.
        """
        if isinstance(X, str):
            raise ValueError(
                "Iterable over raw text documents expected, string object received."
            )

        self._validate_ngram_range()

        analyzer = self.build_analyzer()
        X = self.bloom_bag.transform(analyzer(doc) for doc in X)
        if self.binary:
            X.data.fill(1)
        if self.norm is not None:
            X = normalize(X, norm=self.norm, copy=False)
        return X

    def fit_transform(self, X, y=None):
        """Transform a sequence of documents to a document-term matrix.

        Parameters
        ----------
        X : iterable over raw text documents, length = n_samples
            Samples. Each sample must be a text document (either bytes or
            unicode strings, file name or file object depending on the
            constructor argument) which will be tokenized and hashed.
        y : any
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        X : sparse matrix of shape (n_samples, n_features)
            Document-term matrix.
        """
        return self.fit(X, y).transform(X)

    def _more_tags(self):
        return {"X_types": ["string"]}



