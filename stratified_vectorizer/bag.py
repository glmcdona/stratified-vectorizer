from numbers import Integral, Real
import numpy as np

from fastbloom_rs import BloomFilter, FilterBuilder

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction._hash import _iteritems
from sklearn.utils.validation import check_is_fitted, check_array, FLOAT_DTYPES
from sklearn.utils import _IS_32BIT, IS_PYPY
from sklearn.utils._param_validation import StrOptions, Interval

from scipy.sparse import csr_matrix, coo_matrix

class BloomBagCounting(TransformerMixin, BaseEstimator):
    _parameter_constraints: dict = {
        "n_bags": [Interval(Integral, 1, 15, closed="both")],
        "error_rate": [Interval(Real, 0.0, 1.0, closed="both")],
        "input_type": [StrOptions({"dict", "pair", "string"})],
        "dtype": "no_validation",  # delegate to numpy
    }

    def __init__(
        self,
        n_bags=15,
        *,
        input_type="dict",
        dtype=np.float64,
        error_rate=0.05,
        feature_rank=[],
    ):
        self.dtype = dtype
        self.input_type = input_type
        self.error_rate = error_rate

        self.n_bags = n_bags
        self.feature_rank = feature_rank

        self.counting_bloom_filter = None

        if self.n_bags > len(self.feature_rank):
            self.n_bags = len(self.feature_rank)
            print(
                f"WARNING: n_bags reduced to {self.n_bags} to match feature_rank length"
            )

    def fit(self, X=None, y=None):
        """

        Parameters
        ----------
        X : Ignored
            Not used, present here for API consistency by convention.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            FeatureHasher class instance.
        """
        # repeat input validation for grid search (which calls set_params)
        self._validate_params()

        # Create the counting bloom filter
        # Note: The counting bloom filter has 4 bits for counts, so it can
        # count up to 15. The meaning is:
        #    0: Not in the set, or it falls within multiple buckets
        #    >0: In the set, and the count is the stratified bucket that it
        #        belongs to
        builder = FilterBuilder(len(self.feature_rank), self.error_rate)
        builder.enable_repeat_insert(True)
        self.counting_bloom_filter = builder.build_counting_bloom_filter()  # type: CountingBloomFilter


        # Now build the bloom filter vocabularies
        bucket_size = len(self.feature_rank) / self.n_bags
        
        # First pass: Adding the entries to the bloom filter
        for i, f in enumerate(self.feature_rank):
            feature_bucket = int(i / bucket_size)
            
            # If the feature is already in the counting bloom filter, then
            # it is in multiple buckets and should be ignored
            if f not in self.counting_bloom_filter:
                # Add the feature to the counting bloom filter
                for _ in range(feature_bucket+1):
                    self.counting_bloom_filter.add(f)
        
        # Second pass: Any features with incorrect counts are instead removed
        total_collisions = 0
        collisions = 1
        while collisions > 0:
            collisions = 0
            for i, f in enumerate(self.feature_rank):
                feature_bucket = int(i / bucket_size)
                
                # If the feature is already in the counting bloom filter, then
                # it is in multiple buckets and should be ignored
                if f in self.counting_bloom_filter:
                    # Hackjob. TODO: FIX THIS ONCE COUNT FUNCTION AVAILABLE
                    cur_bucket = -1
                    while f in self.counting_bloom_filter:
                        self.counting_bloom_filter.remove(f)
                        cur_bucket += 1
                    
                    if cur_bucket != feature_bucket:
                        # Collision error, clear this bucket.
                        collisions += 1
                        total_collisions += 1
                    else:
                        # Add the feature back to the counting bloom filter
                        for _ in range(feature_bucket+1):
                            self.counting_bloom_filter.add(f)
            
            print(f"Dropped {collisions} features due to collisions out of {len(self.feature_rank)} features")
        
        print(f"In total dropped {total_collisions} features due to collisions out of {len(self.feature_rank)} features")

        return self

    def get_size_in_bytes(self):
        """
        Returns the size in bytes of this set of BloomBags
        """
        if self.counting_bloom_filter is None:
            return 0
        return len(self.counting_bloom_filter.get_bytes())

    def transform(self, X):
        """Transform a sequence of instances to a scipy.sparse matrix.
        Parameters
        ----------
        X : iterable over iterable over raw features, length = n_samples
            Samples. Each sample must be iterable an (e.g., a list or tuple)
            containing/generating feature names (and optionally values, see
            the input_type constructor argument) which will be hashed.
            raw_X need not support the len function, so it can be the result
            of a generator; n_samples is determined on the fly.
        Returns
        -------
        X : sparse matrix of shape (n_samples, n_features)
            Feature matrix, for use with estimators or further transformers.
        """
        # Process everything as sparse regardless of setting
        X = iter(X)
        if self.input_type == "dict":
            X = (_iteritems(d) for d in X)
        elif self.input_type == "string":
            X = (((f, 1) for f in x) for x in X)

        X_out = []
        for row, x in enumerate(X):
            x = list(x)
            X_out.append([0] * self.n_bags)

            # Iterate values in row
            for v, c in x:
                if self.counting_bloom_filter.contains(v):
                    # Get the bucket number that the feature belongs to
                    # Hackjob. TODO: FIX THIS ONCE COUNT FUNCTION AVAILABLE
                    bucket = -1
                    while(v in self.counting_bloom_filter):
                        self.counting_bloom_filter.remove(v)
                        bucket += 1
                    
                    if bucket >= self.n_bags:
                        # This feature hit a collision error case, ignore it
                        continue
                    
                    X_out[row][bucket] += c
                    for _ in range(bucket+1):
                        self.counting_bloom_filter.add(v)

        return np.array(X_out)

    def _more_tags(self):
        return {"X_types": [self.input_type]}


class BloomStratifiedBag(TransformerMixin, BaseEstimator):
    _parameter_constraints: dict = {
        "n_bags": [Interval(Integral, 1, np.iinfo(np.int32).max, closed="both")],
        "input_type": [StrOptions({"dict", "pair", "string"})],
        "dtype": "no_validation",  # delegate to numpy
        "alternate_sign": ["boolean"],
    }

    def __init__(
        self,
        n_bags=100,
        *,
        input_type="dict",
        dtype=np.float64,
        error_rate=0.05,
        feature_rank=[],
    ):
        self.dtype = dtype
        self.input_type = input_type
        self.error_rate = error_rate

        self.n_bags = n_bags
        self.feature_rank = feature_rank

        if self.n_bags > len(self.feature_rank):
            self.n_bags = len(self.feature_rank)
            print(
                f"WARNING: n_bags reduced to {self.n_bags} to match feature_rank length"
            )

    def fit(self, X=None, y=None):
        """

        Parameters
        ----------
        X : Ignored
            Not used, present here for API consistency by convention.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            FeatureHasher class instance.
        """
        # repeat input validation for grid search (which calls set_params)
        self._validate_params()

        # Now build the bloom filter vocabularies
        self.bloom_filters = []
        bucket_size = len(self.feature_rank) / self.n_bags
        bucket_size_max = int(np.ceil(bucket_size))

        for i in range(self.n_bags):
            # Build the list of bloom filters, matching to each set of features
            # in the bucket
            if i == self.n_bags - 1:
                # Last bucket may have more features
                f = self.feature_rank[int(i * bucket_size) :]
            else:
                f = self.feature_rank[int(i * bucket_size) : int((i + 1) * bucket_size)]

            # Create and fit the bloom filter
            bloom = BloomFilter(bucket_size_max, self.error_rate)
            bloom.add_str_batch(f)

            self.bloom_filters.append(bloom)

        return self

    def get_size_in_bytes(self):
        """
        Returns the size in bytes of this set of BloomBags
        """
        size = 0
        for bloom in self.bloom_filters:
            size += len(bloom.get_bytes())

        return size

    def transform(self, X):
        """Transform a sequence of instances to a scipy.sparse matrix.
        Parameters
        ----------
        X : iterable over iterable over raw features, length = n_samples
            Samples. Each sample must be iterable an (e.g., a list or tuple)
            containing/generating feature names (and optionally values, see
            the input_type constructor argument) which will be hashed.
            raw_X need not support the len function, so it can be the result
            of a generator; n_samples is determined on the fly.
        Returns
        -------
        X : sparse matrix of shape (n_samples, n_features)
            Feature matrix, for use with estimators or further transformers.
        """
        # Process everything as sparse regardless of setting
        X = iter(X)
        if self.input_type == "dict":
            X = (_iteritems(d) for d in X)
        elif self.input_type == "string":
            X = (((f, 1) for f in x) for x in X)

        rows = []
        cols = []
        values = []
        for row, x in enumerate(X):
            x = list(x)

            features = list(map(lambda x: x[0], x))

            for i, bloom in enumerate(self.bloom_filters):
                contains = bloom.contains_str_batch(features)
                for f, c in zip(features, contains):
                    if c:
                        rows.append(row)
                        cols.append(i)
                        values.append(1)

        # Create the sparse matrix
        X = coo_matrix(
            (values, (rows, cols)),
            shape=(row + 1, self.n_bags),
            dtype=self.dtype,
        ).tocsr(copy=False)

        return X

    def _more_tags(self):
        return {"X_types": [self.input_type]}




class StratifiedBag(TransformerMixin, BaseEstimator):
    _parameter_constraints: dict = {
        "n_bags": [Interval(Integral, 1, np.iinfo(np.int32).max, closed="both")],
        "input_type": [StrOptions({"dict", "pair", "string"})],
        "dtype": "no_validation",  # delegate to numpy
        "alternate_sign": ["boolean"],
    }

    def __init__(
        self,
        n_bags=100,
        *,
        input_type="dict",
        dtype=np.float64,
        feature_rank=[],
    ):
        self.dtype = dtype
        self.input_type = input_type

        self.n_bags = n_bags
        self.feature_rank = feature_rank

        if self.n_bags > len(self.feature_rank):
            self.n_bags = len(self.feature_rank)
            print(
                f"WARNING: n_bags reduced to {self.n_bags} to match feature_rank length"
            )

    def fit(self, X=None, y=None):
        """

        Parameters
        ----------
        X : Ignored
            Not used, present here for API consistency by convention.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            FeatureHasher class instance.
        """
        # repeat input validation for grid search (which calls set_params)
        self._validate_params()

        # Now build the bloom filter vocabularies
        self.vocab_to_bag = {}
        bucket_size = len(self.feature_rank) / self.n_bags
        bucket_size_max = int(np.ceil(bucket_size))

        for i in range(self.n_bags):
            # Build the list of bloom filters, matching to each set of features
            # in the bucket
            if i == self.n_bags - 1:
                # Last bucket may have more features
                f = self.feature_rank[int(i * bucket_size) :]
            else:
                f = self.feature_rank[int(i * bucket_size) : int((i + 1) * bucket_size)]

            # Create and fit the bloom filter
            for feature in f:
                self.vocab_to_bag[feature] = i

        return self

    def get_size_in_bytes(self):
        """
        Returns the size in bytes of this set of BloomBags
        """
        size = 0
        for key, value in self.vocab_to_bag.items():
            size += len(key)

        return size

    def transform(self, X):
        """Transform a sequence of instances to a scipy.sparse matrix.
        Parameters
        ----------
        X : iterable over iterable over raw features, length = n_samples
            Samples. Each sample must be iterable an (e.g., a list or tuple)
            containing/generating feature names (and optionally values, see
            the input_type constructor argument) which will be hashed.
            raw_X need not support the len function, so it can be the result
            of a generator; n_samples is determined on the fly.
        Returns
        -------
        X : sparse matrix of shape (n_samples, n_features)
            Feature matrix, for use with estimators or further transformers.
        """
        # Process everything as sparse regardless of setting
        X = iter(X)
        if self.input_type == "dict":
            X = (_iteritems(d) for d in X)
        elif self.input_type == "string":
            X = (((f, 1) for f in x) for x in X)

        coords = []
        values = []
        for row, x in enumerate(X):
            x = list(x)

            for f, v in x:
                if f in self.vocab_to_bag:
                    coords.append((row, self.vocab_to_bag[f]))
                    values.append(v)
        
        # Create the sparse matrix
        X = coo_matrix(
            (values, zip(*coords)),
            shape=(row + 1, self.n_bags),
            dtype=self.dtype,
        ).tocsr(copy=False)

        return X

    def _more_tags(self):
        return {"X_types": [self.input_type]}


