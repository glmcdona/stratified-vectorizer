import pytest
from stratified_vectorizer import StratifiedBagVectorizer, BloomStratifiedBag, BloomBagCounting
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def test_bloom_vectorizer_binary_classification_tokenizer():
    """Test BloomVectorizer with a provided tokenizer"""
    print("\n\n--- Starting test_bloom_vectorizer_binary_classification_tokenizer")
    cats_pos = ["alt.atheism", "sci.space"]
    cats_neg = ["comp.graphics"]
    print("Downloading 20 newsgroups dataset, train")
    train_pos = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"), categories=cats_pos).data
    train_neg = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"), categories=cats_neg).data
    train = train_pos + train_neg
    train_y = [1] * len(train_pos) + [0] * len(train_neg)

    print("Downloading 20 newsgroups dataset, test")
    test_pos = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"), categories=cats_pos).data
    test_neg = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"), categories=cats_neg).data
    test = test_pos + test_neg
    test_y = [1] * len(test_pos) + [0] * len(test_neg)

    n_features = 50000
    n_bags = 10
    pipelines = {
        "StratifiedBagVectorizer_e0": Pipeline(
            [
                ("vectorizer", StratifiedBagVectorizer(
                    tokenizer = lambda x: x.split(' '),
                    n_features = n_features,
                    n_bags = n_bags,
                    error_rate = 0,
                    token_pattern = None,
                )),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        ),
        "StratifiedBagVectorizer_e0_chi-tfidf": Pipeline(
            [
                ("vectorizer", StratifiedBagVectorizer(
                    tokenizer = lambda x: x.split(' '),
                    n_features = n_features,
                    n_bags = n_bags,
                    error_rate = 0,
                    token_pattern = None,
                    ranking_method = "chi-tfidf",
                )),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        ),
        "StratifiedBagVectorizer_e0_chi": Pipeline(
            [
                ("vectorizer", StratifiedBagVectorizer(
                    tokenizer = lambda x: x.split(' '),
                    n_features = n_features,
                    n_bags = n_bags,
                    error_rate = 0,
                    token_pattern = None,
                    ranking_method = "chi",
                )),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        ),
        "StratifiedBagVectorizer_e0.01": Pipeline(
            [
                ("vectorizer", StratifiedBagVectorizer(
                    tokenizer = lambda x: x.split(' '),
                    n_features = n_features,
                    n_bags = n_bags,
                    error_rate = 0.01,
                    token_pattern = None,
                )),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        ),
        "StratifiedBagVectorizerCounting_e0.01": Pipeline(
            [
                ("vectorizer", StratifiedBagVectorizer(
                    tokenizer = lambda x: x.split(' '),
                    n_features = n_features,
                    n_bags = n_bags,
                    error_rate = 0.01,
                    token_pattern = None,
                    stratified_bag_class = BloomBagCounting,
                )),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        ),
        "CountVectorizer": Pipeline(
            [
                ("vectorizer", CountVectorizer(
                    tokenizer = lambda x: x.split(' '),
                    max_features = n_features,
                    token_pattern = None,
                )),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        ),
        "HashVectorizer": Pipeline(
            [
                ("vectorizer", HashingVectorizer(
                    tokenizer = lambda x: x.split(' '),
                    n_features = n_features,
                    token_pattern = None,
                )),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        ),
    }

    for name, pipeline in pipelines.items():
        pipeline.fit(train, train_y)

        # Run against test data for metrics
        y_pred = pipeline.predict(test)

        aucroc = roc_auc_score(test_y, y_pred)
        acc = accuracy_score(test_y, y_pred)
        prec = precision_score(test_y, y_pred)
        rec = recall_score(test_y, y_pred)
        f1 = f1_score(test_y, y_pred)
        
        print(f"\n{name}")
        print("AUCROC: ", aucroc)
        print("Accuracy: ", acc)
        print("Precision: ", prec)
        print("Recall: ", rec)
        print("F1: ", f1)
        # If bloom_bag is an attribute of the vectorizer, then run get_size_in_bytes()
        if hasattr(pipeline.named_steps["vectorizer"], "bloom_bag"):
            print("Size in bytes: ", pipeline.named_steps["vectorizer"].bloom_bag.get_size_in_bytes())

        if name == "BloomVectorizer":
            assert f1 > 0.89
        elif name == "BloomVectorizerCounting":
            assert f1 > 0.81

    pipeline.fit(train, train_y)



def test_bloom_vectorizer_binary_classification_token_pattern():
    """Test BloomVectorizer with a provided token pattern"""
    print("\n\n--- Starting test_bloom_vectorizer_binary_classification_token_pattern")
    cats_pos = ["alt.atheism", "sci.space"]
    cats_neg = ["comp.graphics", "comp.sys.ibm.pc.hardware"]
    print("Downloading 20 newsgroups dataset, train")
    train_pos = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"), categories=cats_pos).data
    train_neg = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"), categories=cats_neg).data
    train = train_pos + train_neg
    train_y = [1] * len(train_pos) + [0] * len(train_neg)

    print("Downloading 20 newsgroups dataset, test")
    test_pos = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"), categories=cats_pos).data
    test_neg = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"), categories=cats_neg).data
    test = test_pos + test_neg
    test_y = [1] * len(test_pos) + [0] * len(test_neg)

    n_features = 50000
    n_bags = 10
    error_rate = 0.01
    pipelines = {
        "BloomVectorizer": Pipeline(
            [
                ("vectorizer", StratifiedBagVectorizer(
                    stop_words='english',
                    n_features = n_features,
                    n_bags = n_bags,
                    error_rate = error_rate,
                )),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        ),
        "BloomVectorizerCounting": Pipeline(
            [
                ("vectorizer", StratifiedBagVectorizer(
                    stop_words='english',
                    n_features = n_features,
                    n_bags = n_bags,
                    error_rate = error_rate,
                    stratified_bag_class = BloomBagCounting,
                )),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        ),
        "CountVectorizer": Pipeline(
            [
                ("vectorizer", CountVectorizer(
                    stop_words='english',
                    max_features = n_features,
                )),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        ),
        "HashVectorizer": Pipeline(
            [
                ("vectorizer", HashingVectorizer(
                    stop_words='english',
                    n_features = n_features,
                )),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        ),
    }

    for name, pipeline in pipelines.items():
        pipeline.fit(train, train_y)

        # Run against test data for metrics
        y_pred = pipeline.predict(test)

        aucroc = roc_auc_score(test_y, y_pred)
        acc = accuracy_score(test_y, y_pred)
        prec = precision_score(test_y, y_pred)
        rec = recall_score(test_y, y_pred)
        f1 = f1_score(test_y, y_pred)
        
        print(f"\n{name}")
        print("AUCROC: ", aucroc)
        print("Accuracy: ", acc)
        print("Precision: ", prec)
        print("Recall: ", rec)
        print("F1: ", f1)
        # If bloom_bag is an attribute of the vectorizer, then run get_size_in_bytes()
        if hasattr(pipeline.named_steps["vectorizer"], "bloom_bag"):
            print("Size in bytes: ", pipeline.named_steps["vectorizer"].bloom_bag.get_size_in_bytes())

        if name == "BloomVectorizer":
            assert f1 > 0.89
        elif name == "BloomVectorizerCounting":
            assert f1 > 0.81

    pipeline.fit(train, train_y)

if __name__ == "__main__":
    #test_bloom_vectorizer_binary_classification_token_pattern()
    test_bloom_vectorizer_binary_classification_tokenizer()
    