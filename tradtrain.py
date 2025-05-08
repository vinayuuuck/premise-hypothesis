import xgboost
import time
import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import (
    VotingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from joblib import dump, load


def build_train_data(trainfile: str = "./data/snli_1.0/snli_1.0_train.jsonl"):
    train_data = pd.read_json(trainfile, lines=True)
    # Remove all rows where gold_label is '-'
    train_data = train_data[train_data["gold_label"] != "-"]
    training_corpus = [
        f"{sentence1} {sentence2}"
        for sentence1, sentence2 in zip(
            train_data["sentence1"], train_data["sentence2"]
        )
    ]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(training_corpus)
    tfidf_premise = vectorizer.transform(train_data["sentence1"].values.astype("U"))
    tfidf_hypothesis = vectorizer.transform(train_data["sentence2"].values.astype("U"))

    train_features = scipy.sparse.hstack((tfidf_premise, tfidf_hypothesis))
    train_labels = train_data["gold_label"].values
    # Preprocess the labels
    label_map = {"entailment": 0, "contradiction": 1, "neutral": 2}
    train_labels = np.array([label_map[label] for label in train_labels])

    missing_rows = train_data[train_data.isnull().any(axis=1)]
    if not missing_rows.empty:
        print("Rows with missing values:")
        print(missing_rows)
    else:
        print("No missing values found.")

    return train_features, train_labels, vectorizer


def build_test_data(
    vectorizer: TfidfVectorizer, testfile: str = "./data/snli_1.0/snli_1.0_test.jsonl"
):
    train_data = pd.read_json(testfile, lines=True)
    # Remove all rows where gold_label is '-'
    train_data = train_data[train_data["gold_label"] != "-"]
    tfidf_premise = vectorizer.transform(train_data["sentence1"].values.astype("U"))
    tfidf_hypothesis = vectorizer.transform(train_data["sentence2"].values.astype("U"))

    test_features = scipy.sparse.hstack((tfidf_premise, tfidf_hypothesis))
    test_labels = train_data["gold_label"].values

    # Preprocess the labels
    label_map = {"entailment": 0, "contradiction": 1, "neutral": 2}
    test_labels = np.array([label_map[label] for label in test_labels])
    return test_features, test_labels


def build_logistic_regression_model(train_features, train_labels):
    param_grid = {
        "C": [0.1, 1.0, 10.0],
        "solver": ["lbfgs", "newton-cg", "saga"],
    }
    grid_search = GridSearchCV(
        LogisticRegression(max_iter=100000),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=4,
    )
    grid_search.fit(train_features, train_labels)
    best_model = grid_search.best_estimator_
    print("Best parameters for Logistic Regression:", grid_search.best_params_)
    return best_model


def build_random_forest_model(train_features, train_labels):
    model = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
    )
    model.fit(train_features, train_labels)
    return model


def build_gradient_boosting_model(train_features, train_labels):
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train_features, train_labels)
    return model


# Testing
def test_model(model, test_features, test_labels):
    start_time = time.time()
    predictions = model.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    end_time = time.time()
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    return accuracy


def main():
    # Load training data
    train_features, train_labels, vectorizer = build_train_data()
    test_features, test_labels = build_test_data(vectorizer)

    # Train models
    logistic_model = build_logistic_regression_model(train_features, train_labels)
    rf_model = build_random_forest_model(train_features, train_labels)
    gb_model = build_gradient_boosting_model(train_features, train_labels)

    test_model(logistic_model, test_features, test_labels)
    test_model(rf_model, test_features, test_labels)
    test_model(gb_model, test_features, test_labels)


if __name__ == "__main__":
    main()
