# 🍄 Mushroom Classification using Machine Learning

Classify mushrooms as **edible** or **poisonous** from their physical characteristics, and compare six
classification algorithms through an interactive Streamlit web app.

**Dataset:** Mushroom Classification (Kaggle / UCI) – 5,644 samples, 21 usable categorical features.

| Planned algorithm (Weka) | scikit-learn implementation |
|---|---|
| Decision Tree (J48) | `DecisionTreeClassifier(criterion="entropy")` |
| Random Forest | `RandomForestClassifier` |
| Naive Bayes | `BernoulliNB` (on one-hot features) |
| SVM (SMO) | `SVC(kernel="rbf")` |
| K-Nearest Neighbors (IBk) | `KNeighborsClassifier(n_neighbors=5)` |
| Logistic Regression | `LogisticRegression` |

## Project structure
```
mushroom-classification/
├── app.py                 # Streamlit web app (dataset • model comparison • live prediction)
├── src/
│   ├── train.py           # trains + evaluates all 6 models, saves them to models/
│   └── features.py        # readable names for the single-letter dataset codes
├── data/mushrooms.csv     # dataset
├── models/                # generated: *.joblib, results.csv, meta.json, feature_importance.csv
├── test_app.py            # smoke test for the web app
├── requirements.txt
└── README.md
```

## Setup & run
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.train                # train models (≈10 s) – regenerates models/

.\.venv\Scripts\python.exe -m streamlit run app.py               # opens http://localhost:8501
```

## Method
1. Drop constant column `veil-type`.
2. One-hot encode all categorical features (inside a scikit-learn `Pipeline`, so no data leakage).
3. 80/20 stratified train/test split (`random_state=42`) + 5-fold stratified cross-validation.
4. Metrics: accuracy, precision, recall, F1, ROC-AUC (poisonous = positive class), training time, confusion matrix.

## Results (test set)
| Model | Accuracy | Recall (poisonous) |
|---|---|---|
| Decision Tree, Random Forest, SVM, KNN, Logistic Regression | 100% | 100% |
| Naive Bayes | 94.2% | 85.4% |

The dataset is almost perfectly separable, so most models are perfect. Naive Bayes is weaker because its
feature-independence assumption does not hold. In this domain a false negative
(poisonous predicted as edible) is the costly error, so recall matters more than accuracy.

> ⚠️ Educational project only. Never eat a wild mushroom based on a model's prediction.
