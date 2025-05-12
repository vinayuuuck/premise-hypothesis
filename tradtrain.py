import time
import numpy as np
import scipy.sparse

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    PrecisionRecallDisplay,
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import learning_curve
from joblib import dump, load
import matplotlib.pyplot as plt


def load_snli_splits():
    snli = load_dataset("snli")
    snli = snli.filter(lambda ex: ex["label"] != -1)
    return snli["train"], snli["validation"], snli["test"]


def build_vectorizer_and_features(train_ds):
    corpus = [f"{p} {h}" for p, h in zip(train_ds["premise"], train_ds["hypothesis"])]
    vec = TfidfVectorizer()
    vec.fit(corpus)

    tfidf_p = vec.transform(train_ds["premise"])
    tfidf_h = vec.transform(train_ds["hypothesis"])
    X = scipy.sparse.hstack((tfidf_p, tfidf_h))
    y = np.array(train_ds["label"])
    return vec, X, y


def transform_features(ds, vec):
    tfidf_p = vec.transform(ds["premise"])
    tfidf_h = vec.transform(ds["hypothesis"])
    X = scipy.sparse.hstack((tfidf_p, tfidf_h))
    y = np.array(ds["label"])
    return X, y


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


def test_model(name, model, X, y):
    start = time.time()
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    print(f"{name} accuracy: {acc:.4f} ({time.time()-start:.2f}s)")
    return acc


def main():
    train_ds, val_ds, test_ds = load_snli_splits()

    vectorizer, X_train, y_train = build_vectorizer_and_features(train_ds)
    X_val, y_val = transform_features(val_ds, vectorizer)
    X_test, y_test = transform_features(test_ds, vectorizer)

    lr = build_logistic(X_train, y_train)
    rf = build_rf(X_train, y_train)
    xg = build_xgb(X_train, y_train)

    test_model("LR (val)", lr, X_val, y_val)
    test_model("RF (val)", rf, X_val, y_val)
    test_model("XGB (val)", xg, X_val, y_val)

    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("xg", xg)], voting="hard", n_jobs=-1
    )
    ensemble.fit(X_train, y_train)
    preds_ens = ensemble.predict(X_val)
    test_model("Ensemble (val)", ensemble, X_val, y_val)

    test_model("Ensemble (test)", ensemble, X_test, y_test)

    dump(vectorizer, "./data/models/vectorizer.joblib")
    dump(lr, "./data/models/logistic.joblib")
    dump(rf, "./data/models/random_forest.joblib")
    dump(xg, "./data/models/xgboost.joblib")
    dump(ensemble, "./data/models/ensemble.joblib")
    print("All models and vectorizer saved.")

    cm = confusion_matrix(y_val, preds_ens, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["entailment", "neutral", "contradiction"]
    )
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("./data/models/confusion_matrix.png")

    train_sizes, train_scores, val_scores = learning_curve(
        ensemble,
        X_train,
        y_train,
        cv=5,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring="accuracy",
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 15))
    plt.plot(train_sizes, train_mean, "o-", color="blue", label="Training score")
    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1,
        color="blue",
    )
    plt.plot(train_sizes, val_mean, "o-", color="green", label="Validation score")
    plt.fill_between(
        train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="green"
    )
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curves")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig("./data/models/learning_curve.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
