"""Train, evaluate and save all six classifiers.

Run from the project root:
    python -m src.train
"""
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.features import TARGET

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "mushrooms.csv"
MODEL_DIR = ROOT / "models"
SEED = 42

# Weka algorithm in the name -> scikit-learn equivalent
MODELS = {
    "Decision Tree (J48)": DecisionTreeClassifier(criterion="entropy", random_state=SEED),  # J48 = C4.5 (entropy)
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
    "Naive Bayes": BernoulliNB(),
    "SVM (SMO)": SVC(kernel="rbf", probability=True, random_state=SEED),  # SMO is an SVM training algorithm
    "K-Nearest Neighbors (IBk)": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000),
}


def model_filename(name: str) -> str:
    return name.split(" (")[0].lower().replace(" ", "_").replace("-", "_") + ".joblib"


def load_data():
    df = pd.read_csv(DATA_PATH)
    # Constant columns (e.g. veil-type) carry no information
    constant = [c for c in df.columns if df[c].nunique() == 1]
    df = df.drop(columns=constant)
    X = df.drop(columns=[TARGET])
    y = (df[TARGET] == "p").astype(int)  # 1 = poisonous (the "positive" class)
    return X, y, constant


def build_pipeline(model, columns):
    pre = ColumnTransformer([("onehot", OneHotEncoder(handle_unknown="ignore"), columns)])
    return Pipeline([("prep", pre), ("clf", model)])


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    X, y, dropped = load_data()
    cols = list(X.columns)
    print(f"Rows: {len(X)} | Features: {len(cols)} | Dropped constant: {dropped}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results, confusions = [], {}
    for name, model in MODELS.items():
        pipe = build_pipeline(model, cols)
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        cv_scores = cross_val_score(build_pipeline(model, cols), X, y, cv=cv, scoring="accuracy")

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall (poisonous)": recall_score(y_test, pred, zero_division=0),
            "F1-score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, proba),
            "CV Accuracy (5-fold)": cv_scores.mean(),
            "CV Std": cv_scores.std(),
            "Train time (s)": train_time,
        })
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        confusions[name] = {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
        joblib.dump(pipe, MODEL_DIR / model_filename(name))
        print(f"{name:28s} acc={results[-1]['Accuracy']:.4f}  cv={cv_scores.mean():.4f}  FN={fn}")

    res_df = pd.DataFrame(results).sort_values(["Accuracy", "F1-score"], ascending=False)
    res_df.to_csv(MODEL_DIR / "results.csv", index=False)

    # Random Forest feature importance, summed back to the original columns
    rf = joblib.load(MODEL_DIR / model_filename("Random Forest"))
    names = rf.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names)
    agg = {c: float(imp[[n for n in names if n.startswith(f"onehot__{c}_")]].sum()) for c in cols}
    pd.Series(agg).sort_values(ascending=False).rename("importance").to_csv(
        MODEL_DIR / "feature_importance.csv", index_label="feature")

    meta = {"n_rows": int(len(X)), "n_features": len(cols), "dropped": dropped,
            "test_size": 0.2, "seed": SEED, "confusion": confusions, "columns": cols,
            "model_files": {n: model_filename(n) for n in MODELS}}
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print("\n", res_df.round(4).to_string(index=False))
    print(f"\nSaved models and metrics to {MODEL_DIR}")


if __name__ == "__main__":
    main()
