import time
import numpy as np
import scipy.sparse

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load


# 1. Load SNLI splits
def load_snli_splits():
    snli = load_dataset("snli")
    snli = snli.filter(lambda ex: ex["label"] != -1)
    return snli["train"], snli["validation"], snli["test"]


# 2. Build TF–IDF vectorizer & features
def build_vectorizer_and_features(train_ds):
    corpus = [f"{p} {h}" for p, h in zip(train_ds["premise"], train_ds["hypothesis"])]
    vec = TfidfVectorizer()
    vec.fit(corpus)

    tfidf_p = vec.transform(train_ds["premise"])
    tfidf_h = vec.transform(train_ds["hypothesis"])
    X = scipy.sparse.hstack((tfidf_p, tfidf_h))
    y = np.array(train_ds["label"])
    return vec, X, y


# 3. Transform any split
def transform_features(ds, vec):
    tfidf_p = vec.transform(ds["premise"])
    tfidf_h = vec.transform(ds["hypothesis"])
    X = scipy.sparse.hstack((tfidf_p, tfidf_h))
    y = np.array(ds["label"])
    return X, y


# 4. Model builders
def build_logistic(X, y):
    param_grid = {"C": [0.1, 1.0, 10.0], "solver": ["lbfgs", "newton-cg", "saga"]}
    grid = GridSearchCV(
        LogisticRegression(max_iter=100_000),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2,
    )
    grid.fit(X, y)
    print("LR best params:", grid.best_params_)
    return grid.best_estimator_


def build_rf(X, y):
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
    )
    rf.fit(X, y)
    return rf


def build_xgb(X, y):
    xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    xgb.fit(X, y)
    return xgb


# 5. Evaluation helper
def test_model(name, model, X, y):
    start = time.time()
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    print(f"{name} accuracy: {acc:.4f} ({time.time()-start:.2f}s)")
    return acc


def main():
    # Load data
    train_ds, val_ds, test_ds = load_snli_splits()

    # Build features
    vectorizer, X_train, y_train = build_vectorizer_and_features(train_ds)
    X_val, y_val = transform_features(val_ds, vectorizer)
    X_test, y_test = transform_features(test_ds, vectorizer)

    # Train
    lr = build_logistic(X_train, y_train)
    rf = build_rf(X_train, y_train)
    xg = build_xgb(X_train, y_train)

    # Validate
    test_model("LR (val)", lr, X_val, y_val)
    test_model("RF (val)", rf, X_val, y_val)
    test_model("XGB (val)", xg, X_val, y_val)

    # Ensemble
    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("xg", xg)], voting="hard", n_jobs=-1
    )
    ensemble.fit(X_train, y_train)
    test_model("Ensemble (val)", ensemble, X_val, y_val)

    # Final test
    test_model("Ensemble (test)", ensemble, X_test, y_test)

    # Save vectorizer and models
    dump(vectorizer, "./data/models/vectorizer.joblib")
    dump(lr, "./data/models/logistic.joblib")
    dump(rf, "./data/models/random_forest.joblib")
    dump(xg, "./data/models/xgboost.joblib")
    dump(ensemble, "./data/models/ensemble.joblib")
    print("All models and vectorizer saved.")


if __name__ == "__main__":
    main()
