"""Streamlit web app: Mushroom Classification using Machine Learning.

Run:
    .venv\Scripts\python.exe -m streamlit run app.py
"""

import json

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from src.features import CLASS_LABELS, FEATURES, TARGET
from src.train import DATA_PATH, MODEL_DIR


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Mushroom Classifier",
    page_icon="🍄",
    layout="wide",
)


# ============================================================
# HELP / EXPLANATIONS
# ============================================================

METRIC_HELP = {
    "Accuracy": (
        "Accuracy is the percentage of all mushrooms that the model "
        "classified correctly. Formula: (Correct Predictions / "
        "Total Predictions) × 100."
    ),

    "Precision": (
        "Precision tells us, out of all mushrooms predicted as "
        "poisonous, how many were actually poisonous. High precision "
        "means fewer edible mushrooms are incorrectly labelled poisonous."
    ),

    "Recall (poisonous)": (
        "Recall tells us, out of all actually poisonous mushrooms, "
        "how many the model successfully identified as poisonous. "
        "For this project, high recall is especially important because "
        "missing a poisonous mushroom is the more dangerous error."
    ),

    "F1-score": (
        "F1-score is the harmonic mean of Precision and Recall. "
        "It gives a single score that balances both measures. "
        "A higher F1-score means a better balance between precision "
        "and recall."
    ),

    "ROC-AUC": (
        "ROC-AUC measures how well the model separates edible and "
        "poisonous mushrooms across different classification thresholds. "
        "1.0 is excellent separation, while 0.5 is roughly equivalent "
        "to random guessing."
    ),

    "CV Accuracy (5-fold)": (
        "5-fold cross-validation accuracy. The training data is divided "
        "into 5 parts. The model trains on 4 parts and validates on the "
        "remaining part, repeating this 5 times. The reported value is "
        "the average accuracy across those 5 runs."
    ),

    "CV Std": (
        "Standard deviation of the 5 cross-validation accuracy scores. "
        "A smaller value means the model's performance was more consistent "
        "across the different folds."
    ),

    "Train time (s)": (
        "The amount of time, in seconds, required to train the model."
    ),
}


# ============================================================
# CACHED LOADERS
# ============================================================

@st.cache_data
def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_results():
    results = pd.read_csv(MODEL_DIR / "results.csv")
    importance = pd.read_csv(MODEL_DIR / "feature_importance.csv")
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    return results, importance, meta


@st.cache_resource
def load_model(filename: str):
    return joblib.load(MODEL_DIR / filename)


def pretty(col: str) -> str:
    return FEATURES.get(col, (col, {}))[0]


# ============================================================
# PROBABILITY HELPER
# ============================================================

def get_poison_probability(model, X):
    """
    Return the probability that the mushroom is poisonous.

    Uses model.classes_ so that we do not blindly assume
    predict_proba()[0, 1] always represents poisonous.
    """

    probabilities = model.predict_proba(X)[0]
    classes = list(model.classes_)

    poisonous_index = None

    for i, cls in enumerate(classes):

        cls_str = str(cls).lower()

        if cls_str in {
            "p",
            "poisonous",
            "1",
            "true",
        }:
            poisonous_index = i
            break

    if poisonous_index is None:

        if 1 in classes:
            poisonous_index = classes.index(1)
        else:
            poisonous_index = 1

    return float(probabilities[poisonous_index])


# ============================================================
# CHECK MODELS
# ============================================================

if not (MODEL_DIR / "results.csv").exists():

    st.error(
        "Models not found. Run `python -m src.train` first, "
        "then restart the app."
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

df = load_raw()

results, importance, meta = load_results()


# ============================================================
# HEADER
# ============================================================

st.title("🍄 Mushroom Classification using Machine Learning")

st.caption(
    "Classify mushrooms as **edible** or **poisonous** from their "
    "physical characteristics, and compare six classification algorithms."
)

st.warning(
    "Educational project only. Never eat a wild mushroom based on "
    "a model's prediction.",
    icon="⚠️",
)


tab_data, tab_models, tab_predict = st.tabs(
    [
        "📊 Dataset",
        "🏆 Model Comparison",
        "🔮 Predict",
    ]
)


# ============================================================
# TAB 1 — DATASET
# ============================================================

with tab_data:

    st.subheader("Dataset overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Samples",
        f"{len(df):,}",
        help=(
            "The total number of mushroom records in the dataset. "
            "Each row represents one mushroom."
        ),
    )

    c2.metric(
        "Features",
        meta["n_features"],
        help=(
            "The number of input characteristics used by the machine "
            "learning models to classify a mushroom."
        ),
    )

    c3.metric(
        "Edible",
        f"{(df[TARGET] == 'e').sum():,}",
        help=(
            "Number of mushrooms in the dataset labelled as edible."
        ),
    )

    c4.metric(
        "Poisonous",
        f"{(df[TARGET] == 'p').sum():,}",
        help=(
            "Number of mushrooms in the dataset labelled as poisonous."
        ),
    )

    dropped = meta.get("dropped", [])

    st.write(
        "No missing values. Constant column(s) dropped before training: "
        f"`{', '.join(dropped) if dropped else 'None'}`."
    )

    left, right = st.columns([1, 1])


    # --------------------------------------------------------
    # FEATURE EXPLORATION
    # --------------------------------------------------------

    with left:

        st.subheader("Explore a feature")

        col = st.selectbox(
            "Feature",
            [
                c
                for c in df.columns
                if c != TARGET
            ],
            format_func=pretty,
            help=(
                "Choose one mushroom characteristic to see how its "
                "values are distributed between edible and poisonous "
                "mushrooms."
            ),
        )

        mapping = FEATURES.get(
            col,
            (col, {})
        )[1]

        plot_df = (
            df.assign(
                Feature=df[col]
                .map(mapping)
                .fillna(df[col]),

                Class=df[TARGET]
                .map(CLASS_LABELS),
            )
            .groupby(
                [
                    "Feature",
                    "Class",
                ]
            )
            .size()
            .reset_index(
                name="Count"
            )
        )

        chart = (
            alt.Chart(plot_df)
            .mark_bar()
            .encode(

                x=alt.X(
                    "Feature:N",
                    sort="-y",
                    title=pretty(col),
                ),

                y=alt.Y(
                    "Count:Q",
                    title="Count",
                ),

                color=alt.Color(
                    "Class:N",
                    scale=alt.Scale(
                        domain=[
                            "Edible",
                            "Poisonous",
                        ],
                        range=[
                            "#2e9e5b",
                            "#d6453d",
                        ],
                    ),
                ),

                xOffset="Class:N",

                tooltip=[
                    "Feature",
                    "Class",
                    "Count",
                ],
            )
            .properties(
                height=350
            )
        )

        st.altair_chart(
            chart,
            width="stretch",
        )


    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    with right:

        st.subheader("Most informative features")

        st.caption(
            "Feature importance shows how useful each feature was "
            "for the Random Forest model when making predictions."
        )

        top = (
            importance
            .head(10)
            .assign(
                feature=lambda d:
                    d["feature"].map(pretty)
            )
        )

        importance_chart = (
            alt.Chart(top)
            .mark_bar(
                color="#6c5ce7"
            )
            .encode(

                x=alt.X(
                    "importance:Q",
                    title="Random Forest importance",
                ),

                y=alt.Y(
                    "feature:N",
                    sort="-x",
                    title=None,
                ),

                tooltip=[
                    "feature",

                    alt.Tooltip(
                        "importance:Q",
                        format=".3f",
                    ),
                ],
            )
            .properties(
                height=350
            )
        )

        st.altair_chart(
            importance_chart,
            width="stretch",
        )


    # --------------------------------------------------------
    # RAW DATA
    # --------------------------------------------------------

    with st.expander("View raw data"):

        st.dataframe(
            df,
            width="stretch",
            height=300,
        )


# ============================================================
# TAB 2 — MODEL COMPARISON
# ============================================================

with tab_models:

    st.subheader(
        "Test-set performance (80/20 stratified split)"
    )

    st.caption(
        "The dataset is divided into 80% training data and 20% "
        "testing data. The test data is used to evaluate models "
        "on mushrooms they did not train on."
    )


    # --------------------------------------------------------
    # METRIC EXPLANATIONS
    # --------------------------------------------------------

    st.subheader("What do these metrics mean?")

    st.caption(
        "Hover over the ⓘ icons next to each metric for an explanation."
    )

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            "Accuracy",
            f"{results['Accuracy'].max():.2%}",
            help=METRIC_HELP["Accuracy"],
        )

        st.metric(
            "Precision",
            f"{results['Precision'].max():.2%}",
            help=METRIC_HELP["Precision"],
        )

        st.metric(
            "Recall",
            f"{results['Recall (poisonous)'].max():.2%}",
            help=METRIC_HELP["Recall (poisonous)"],
        )

    with metric2:

        st.metric(
            "F1-score",
            f"{results['F1-score'].max():.2%}",
            help=METRIC_HELP["F1-score"],
        )

        st.metric(
            "ROC-AUC",
            f"{results['ROC-AUC'].max():.2%}",
            help=METRIC_HELP["ROC-AUC"],
        )

        st.metric(
            "5-fold CV Accuracy",
            f"{results['CV Accuracy (5-fold)'].max():.2%}",
            help=METRIC_HELP["CV Accuracy (5-fold)"],
        )

    with metric3:

        st.metric(
            "CV Std",
            f"{results['CV Std'].min():.4f}",
            help=METRIC_HELP["CV Std"],
        )

        st.metric(
            "Fastest Train Time",
            f"{results['Train time (s)'].min():.3f}s",
            help=METRIC_HELP["Train time (s)"],
        )

        st.info(
            "The values shown above are the best/max values across "
            "the models for that metric. Use the table below to "
            "compare every model."
        )


    # --------------------------------------------------------
    # PERFORMANCE TABLE
    # --------------------------------------------------------

    st.subheader("Model performance")

    pct_cols = [
        "Accuracy",
        "Precision",
        "Recall (poisonous)",
        "F1-score",
        "ROC-AUC",
        "CV Accuracy (5-fold)",
    ]

    formatted_results = results.copy()

    for c in pct_cols:

        if c in formatted_results.columns:

            formatted_results[c] = formatted_results[c].map(
                lambda x: f"{x:.2%}"
            )

    if "CV Std" in formatted_results.columns:

        formatted_results["CV Std"] = formatted_results[
            "CV Std"
        ].map(
            lambda x: f"{x:.4f}"
        )

    if "Train time (s)" in formatted_results.columns:

        formatted_results["Train time (s)"] = formatted_results[
            "Train time (s)"
        ].map(
            lambda x: f"{x:.3f}"
        )


    # Column tooltips.
    column_config = {

        "Accuracy": st.column_config.TextColumn(
            "Accuracy",
            help=METRIC_HELP["Accuracy"],
        ),

        "Precision": st.column_config.TextColumn(
            "Precision",
            help=METRIC_HELP["Precision"],
        ),

        "Recall (poisonous)": st.column_config.TextColumn(
            "Recall (poisonous)",
            help=METRIC_HELP["Recall (poisonous)"],
        ),

        "F1-score": st.column_config.TextColumn(
            "F1-score",
            help=METRIC_HELP["F1-score"],
        ),

        "ROC-AUC": st.column_config.TextColumn(
            "ROC-AUC",
            help=METRIC_HELP["ROC-AUC"],
        ),

        "CV Accuracy (5-fold)": st.column_config.TextColumn(
            "CV Accuracy (5-fold)",
            help=METRIC_HELP["CV Accuracy (5-fold)"],
        ),

        "CV Std": st.column_config.TextColumn(
            "CV Std",
            help=METRIC_HELP["CV Std"],
        ),

        "Train time (s)": st.column_config.TextColumn(
            "Train time (s)",
            help=METRIC_HELP["Train time (s)"],
        ),

        "Model": st.column_config.TextColumn(
            "Model",
            help=(
                "The machine learning algorithm used to classify "
                "the mushrooms."
            ),
        ),
    }


    st.dataframe(
        formatted_results,
        column_config=column_config,
        width="stretch",
        hide_index=True,
    )


    # --------------------------------------------------------
    # MODEL COMPARISON CHART
    # --------------------------------------------------------

    metric = st.selectbox(
        "Compare models by",
        pct_cols,
        index=0,
        help=(
            "Choose which evaluation metric should be displayed "
            "in the chart below."
        ),
    )

    bar = (
        alt.Chart(results)
        .mark_bar()
        .encode(

            x=alt.X(
                "Model:N",
                sort="-y",
                axis=alt.Axis(
                    labelAngle=-30
                ),
            ),

            y=alt.Y(
                f"{metric}:Q",
                scale=alt.Scale(
                    domain=[
                        max(
                            0,
                            results[metric].min() - 0.05,
                        ),
                        1.0,
                    ]
                ),
            ),

            color=alt.Color(
                "Model:N",
                legend=None,
            ),

            tooltip=[
                "Model",

                alt.Tooltip(
                    f"{metric}:Q",
                    format=".2%",
                ),
            ],
        )
        .properties(
            height=350
        )
    )

    st.altair_chart(
        bar,
        width="stretch",
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader("Confusion matrix")

    st.caption(
        "A confusion matrix shows exactly how many edible and "
        "poisonous mushrooms were classified correctly or incorrectly."
    )

    model_name = st.selectbox(
        "Model",
        results["Model"].tolist(),
        key="cm_model",
        help=(
            "Choose a model to see its confusion matrix."
        ),
    )

    cm = meta["confusion"][model_name]

    # Explanation of TP/TN/FP/FN.

    with st.expander("What do TN, FP, FN and TP mean?"):

        st.markdown(
            """
            **TN — True Negative:**  
            An edible mushroom correctly predicted as edible.

            **FP — False Positive:**  
            An edible mushroom incorrectly predicted as poisonous.

            **FN — False Negative:**  
            A poisonous mushroom incorrectly predicted as edible.
            This is particularly important in this project.

            **TP — True Positive:**  
            A poisonous mushroom correctly predicted as poisonous.
            """
        )

    cm_df = pd.DataFrame(
        [
            {
                "Actual": "Edible",
                "Predicted": "Edible",
                "Count": cm["TN"],
            },

            {
                "Actual": "Edible",
                "Predicted": "Poisonous",
                "Count": cm["FP"],
            },

            {
                "Actual": "Poisonous",
                "Predicted": "Edible",
                "Count": cm["FN"],
            },

            {
                "Actual": "Poisonous",
                "Predicted": "Poisonous",
                "Count": cm["TP"],
            },
        ]
    )

    base = alt.Chart(cm_df).encode(

        x=alt.X(
            "Predicted:N",
            sort=[
                "Edible",
                "Poisonous",
            ],
        ),

        y=alt.Y(
            "Actual:N",
            sort=[
                "Edible",
                "Poisonous",
            ],
        ),
    )

    heat = base.mark_rect().encode(

        color=alt.Color(
            "Count:Q",
            scale=alt.Scale(
                scheme="blues"
            ),
            legend=None,
        )
    )

    text = base.mark_text(
        fontSize=22,
        fontWeight="bold",
    ).encode(

        text="Count:Q",

        color=alt.condition(
            alt.datum.Count
            > cm_df["Count"].max() / 2,

            alt.value("white"),
            alt.value("black"),
        ),
    )

    st.altair_chart(
        (heat + text).properties(
            width=380,
            height=320,
        )
    )


    if cm["FN"] > 0:

        st.error(
            f"{cm['FN']} poisonous mushroom(s) were wrongly "
            f"labelled **edible** — the most dangerous kind of error."
        )

    else:

        st.success(
            "No poisonous mushrooms were misclassified as edible."
        )


    st.info(
        "**Observation:** this dataset is almost perfectly separable "
        "(for example, odor and spore-print colour are very strong "
        "signals), so most models reach close to 100%. Naive Bayes "
        "can perform differently because it assumes features are "
        "independent."
    )


# ============================================================
# TAB 3 — PREDICTION
# ============================================================

with tab_predict:

    st.subheader("Describe a mushroom")

    st.caption(
        "Select the physical characteristics of the mushroom. "
        "The selected machine learning model will then estimate "
        "the probability that it is poisonous."
    )

    chosen_model = st.selectbox(
        "Algorithm",
        results["Model"].tolist(),
        key="pred_model",
        help=(
            "Choose which trained machine learning algorithm "
            "should make the prediction."
        ),
    )

    cols_ui = st.columns(3)

    sample = {}


    for i, col in enumerate(meta["columns"]):

        label, mapping = FEATURES[col]

        observed = sorted(
            df[col].unique()
        )

        with cols_ui[i % 3]:

            sample[col] = st.selectbox(

                label,

                observed,

                format_func=lambda c, m=mapping:
                    f"{m.get(c, c)} ({c})",

                key=f"in_{col}",
            )


    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    if st.button(
        "Classify mushroom",
        type="primary",
        help=(
            "Use the selected model and mushroom characteristics "
            "to calculate the prediction."
        ),
    ):

        model = load_model(
            meta["model_files"][chosen_model]
        )

        X_new = pd.DataFrame(
            [sample]
        )[meta["columns"]]

        # Get poisonous probability.
        p_poison = get_poison_probability(
            model,
            X_new,
        )

        # Probability of edible is the complement.
        p_edible = 1.0 - p_poison

        # Confidence is the probability of the predicted class.
        confidence = max(
            p_edible,
            p_poison,
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        if p_poison >= 0.5:

            st.error(
                f"### ☠️ Predicted: POISONOUS\n"
                f"Confidence: {confidence:.1%}"
            )

        else:

            st.success(
                f"### ✅ Predicted: EDIBLE\n"
                f"Confidence: {confidence:.1%}"
            )


        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        st.subheader("Prediction probability")

        prob_col1, prob_col2 = st.columns(2)

        with prob_col1:

            st.metric(
                "🍄 Edible probability",
                f"{p_edible:.2%}",
                help=(
                    "The model's estimated probability that the "
                    "selected mushroom belongs to the edible class. "
                    "This is a model estimate, not a guarantee."
                ),
            )

        with prob_col2:

            st.metric(
                "☠️ Poisonous probability",
                f"{p_poison:.2%}",
                help=(
                    "The model's estimated probability that the "
                    "selected mushroom belongs to the poisonous class. "
                    "This is a model estimate, not a guarantee."
                ),
            )


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        st.metric(
            "Model confidence",
            f"{confidence:.2%}",
            help=(
                "Confidence here means the probability assigned by "
                "the model to its predicted class. For example, if "
                "poisonous probability is 98%, the predicted class "
                "is poisonous and the displayed confidence is 98%."
            ),
        )


        # Probability bar.

        st.progress(
            p_poison,
            text=(
                f"Probability poisonous: "
                f"{p_poison:.1%}"
            ),
        )

        st.caption(
            "⚠️ Model probability is not a guarantee of safety. "
            "Never use this application to decide whether a wild "
            "mushroom is safe to eat."
        )


        # ----------------------------------------------------
        # ALL MODEL PREDICTIONS
        # ----------------------------------------------------

        with st.expander(
            "What does every model say?",
            expanded=True,
        ):

            st.caption(
                "Compare the probability estimates produced by "
                "each trained algorithm."
            )

            rows = []


            for name, fname in meta["model_files"].items():

                current_model = load_model(
                    fname
                )

                p = get_poison_probability(
                    current_model,
                    X_new,
                )

                edible_probability = 1 - p

                model_confidence = max(
                    p,
                    edible_probability,
                )

                rows.append(
                    {
                        "Model": name,

                        "Prediction": (
                            "☠️ Poisonous"
                            if p >= 0.5
                            else "✅ Edible"
                        ),

                        "P(Edible)": (
                            f"{edible_probability:.2%}"
                        ),

                        "P(Poisonous)": (
                            f"{p:.2%}"
                        ),

                        "Confidence": (
                            f"{model_confidence:.2%}"
                        ),
                    }
                )


            prediction_df = pd.DataFrame(
                rows
            )


            prediction_column_config = {

                "Model": st.column_config.TextColumn(
                    "Model",
                    help=(
                        "The machine learning algorithm making "
                        "the prediction."
                    ),
                ),

                "Prediction": st.column_config.TextColumn(
                    "Prediction",
                    help=(
                        "The class with the higher predicted "
                        "probability: edible or poisonous."
                    ),
                ),

                "P(Edible)": st.column_config.TextColumn(
                    "P(Edible)",
                    help=(
                        "Estimated probability that the mushroom "
                        "is edible."
                    ),
                ),

                "P(Poisonous)": st.column_config.TextColumn(
                    "P(Poisonous)",
                    help=(
                        "Estimated probability that the mushroom "
                        "is poisonous."
                    ),
                ),

                "Confidence": st.column_config.TextColumn(
                    "Confidence",
                    help=(
                        "Probability assigned to the predicted "
                        "class. It is the larger of the edible "
                        "and poisonous probabilities."
                    ),
                ),
            }


            st.dataframe(
                prediction_df,
                column_config=prediction_column_config,
                hide_index=True,
                width="stretch",
            )


            # ------------------------------------------------
            # MODEL PROBABILITY CHART
            # ------------------------------------------------

            chart_df = pd.DataFrame(
                [
                    {
                        "Model": name,

                        "Probability": (
                            get_poison_probability(
                                load_model(fname),
                                X_new,
                            )
                        ),
                    }

                    for name, fname
                    in meta["model_files"].items()
                ]
            )


            probability_chart = (
                alt.Chart(chart_df)

                .mark_bar()

                .encode(

                    x=alt.X(
                        "Model:N",
                        sort="-y",
                        axis=alt.Axis(
                            labelAngle=-30
                        ),
                    ),

                    y=alt.Y(
                        "Probability:Q",
                        scale=alt.Scale(
                            domain=[
                                0,
                                1,
                            ]
                        ),
                        title=(
                            "Probability of being poisonous"
                        ),
                    ),

                    color=alt.Color(
                        "Model:N",
                        legend=None,
                    ),

                    tooltip=[
                        "Model",

                        alt.Tooltip(
                            "Probability:Q",
                            format=".2%",
                        ),
                    ],
                )

                .properties(
                    height=350
                )
            )


            st.altair_chart(
                probability_chart,
                width="stretch",
            )