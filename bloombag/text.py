
from functools import partial
import math
from numbers import Integral, Real

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import _VectorizerMixin, CountVectorizer, TfidfVectorizer
from sklearn.utils._param_validation import StrOptions, Interval
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .bag import BloomStratifiedBag, BloomBagCounting, StratifiedBag




class StratifiedBagVectorizer(
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
        error_rate=0,
        feature_rank=None,
        ranking_method="TfidfVectorizer",
        stratified_bag_class=None,
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
        self.stratified_bag_class = stratified_bag_class
        self.ngram_range = ngram_range
        self.binary = binary
        self.norm = norm
        self.dtype = dtype
        self.stratified_bag = None
        self.ranking_method = ranking_method

        if self.error_rate == 0 and self.stratified_bag_class is not None:
            if self.stratified_bag_class != StratifiedBag:
                raise ValueError(
                    "When error_rate is 0, stratified_bag_class must be StratifiedBag"
                )
        elif self.error_rate == 0:
            self.stratified_bag_class = StratifiedBag
        elif self.stratified_bag_class is None:
            self.stratified_bag_class = BloomStratifiedBag
        
        # Add error rate if it's a bloom bag base
        if self.stratified_bag_class in [BloomStratifiedBag, BloomBagCounting]:
            self.stratified_bag_class = partial(
                self.stratified_bag_class, error_rate=self.error_rate
            )

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

        if self.ranking_method == "CountVectorizer":
            pipeline = Pipeline(
                [
                    (
                        "features",
                        CountVectorizer(
                            tokenizer = self.tokenizer,
                            max_features = n_features,
                            token_pattern = self.token_pattern,
                            stop_words = self.stop_words,
                            ngram_range = self.ngram_range,
                            analyzer = self.analyzer,
                            lowercase = self.lowercase,
                            preprocessor = self.preprocessor,
                            strip_accents = self.strip_accents,
                            decode_error = self.decode_error,
                            encoding = self.encoding,
                        ),
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
        elif self.ranking_method == "TfidfVectorizer":
            pipeline = Pipeline(
                [
                    (
                        "features",
                        TfidfVectorizer(
                            tokenizer = self.tokenizer,
                            max_features = n_features,
                            token_pattern = self.token_pattern,
                            stop_words = self.stop_words,
                            ngram_range = self.ngram_range,
                            analyzer = self.analyzer,
                            lowercase = self.lowercase,
                            preprocessor = self.preprocessor,
                            strip_accents = self.strip_accents,
                            decode_error = self.decode_error,
                            encoding = self.encoding,
                        ),
                    ),
                    ("classifier", LogisticRegression(solver="lbfgs", max_iter=1000)),
                ]
            )
            pipeline.fit(X, y)

            weights = pipeline.named_steps["classifier"].coef_
            # Apply the scaling of the TF-IDF weights
            _idf_diag = pipeline.named_steps["features"]._tfidf.idf_
            weights = weights * _idf_diag

            feature_names = pipeline.named_steps["features"].vocabulary_
            feature_names = [
                feature
                for feature, index in sorted(feature_names.items(), key=lambda x: x[1])
            ]
            ordered_features = [
                feature
                for weight, feature in sorted(zip(weights[0], feature_names), reverse=True)
            ]
        elif self.ranking_method == "chi" or self.ranking_method == "chi-tfidf":
            # Iterate over the training data to count the number of occurrences of each feature
            if self.tokenizer is None:
                raise ValueError(
                    "Tokenizer must be set when using the 'chi' ranking method."
                )
            
            vocab = {}
            y_mean = 0
            for x, Y in zip(X, y):
                y_mean += Y
                for token in self.tokenizer(x):
                    if token not in vocab:
                        vocab[token] = {"cnt": 0, "pos": 0}
                    
                    vocab[token]["cnt"] += 1
                    if Y == 1:
                        vocab[token]["pos"] += 1
            
            y_mean = y_mean / len(y)
            
            # Compute the chi score for each feature
            total = (x["cnt"] for x in vocab.values())
            observed = (x["pos"] for x in vocab.values())
            expected = (x["cnt"] * y_mean for x in vocab.values())
            

            if self.ranking_method == "chi-tfidf":
                # Now compute the TF-IDF weights
                
                # Compute the IDF weights
                tfidf = {}
                for token in vocab:
                    # +1's added to matches sklearn's implementation smooth
                    # default implementation.
                    tfidf[token] = (np.log((len(y) + 1) / (vocab[token]["cnt"] + 1)) + 1) * (vocab[token]["cnt"])
                
                # Compute the TF-IDF weights
                rank = tuple(
                    map(lambda token, t, o, e: tfidf[token] * (o - e) / t, vocab.keys(), total, observed, expected)
                )
            else:
                # No TF-IDF scaling
                rank = tuple(
                    map(lambda t, o, e: (o - e) / t, total, observed, expected)
                )


            # Order the feature_names array by the feature weight
            feature_names = list(vocab.keys())
            feature_ranks = list(rank)
            feature_ranks, ordered_features = zip(
                *sorted(zip(feature_ranks, feature_names))
            )
                
        else:
            raise ValueError(
                "Invalid ranking method. Valid options are 'CountVectorizer' and 'TfidfVectorizer'."
            )

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

        # Build the sratified bag of words
        self.stratified_bag = self.stratified_bag_class(
            n_bags=self.n_bags,
            input_type="string",
            feature_rank=self.ranked_features,
        )

        analyzer = self.build_analyzer()
        self.stratified_bag.fit((analyzer(doc) for doc in X), y=y)
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
        X = self.stratified_bag.transform(analyzer(doc) for doc in X)
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

    def get_size_in_bytes(self):
        if self.stratified_bag is None:
            raise ValueError("Bloom bag is not fitted yet.")
        return self.stratified_bag.get_size_in_bytes()

    def _more_tags(self):
        return {"X_types": ["string"]}

