"""Streamlit web app: Mushroom Classification using Machine Learning.

Run:  streamlit run app.py
"""
import json
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from src.features import CLASS_LABELS, FEATURES, TARGET
from src.train import DATA_PATH, MODEL_DIR

st.set_page_config(page_title="Mushroom Classifier", page_icon="🍄", layout="wide")


# ---------- cached loaders ----------
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


if not (MODEL_DIR / "results.csv").exists():
    st.error("Models not found. Run `python -m src.train` first, then restart the app.")
    st.stop()

df = load_raw()
results, importance, meta = load_results()

# ---------- header ----------
st.title("🍄 Mushroom Classification using Machine Learning")
st.caption("Classify mushrooms as **edible** or **poisonous** from their physical characteristics, "
           "and compare six classification algorithms.")
st.warning("Educational project only. Never eat a wild mushroom based on a model's prediction.", icon="⚠️")

tab_data, tab_models, tab_predict = st.tabs(["📊 Dataset", "🏆 Model Comparison", "🔮 Predict"])

# ---------- TAB 1: dataset ----------
with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples", f"{len(df):,}")
    c2.metric("Features", meta["n_features"])
    c3.metric("Edible", f"{(df[TARGET] == 'e').sum():,}")
    c4.metric("Poisonous", f"{(df[TARGET] == 'p').sum():,}")
    st.write(f"No missing values. Constant column(s) dropped before training: `{', '.join(meta['dropped'])}`.")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Explore a feature")
        col = st.selectbox("Feature", [c for c in df.columns if c != TARGET], format_func=pretty)
        mapping = FEATURES.get(col, (col, {}))[1]
        plot_df = (df.assign(Feature=df[col].map(mapping).fillna(df[col]),
                             Class=df[TARGET].map(CLASS_LABELS))
                     .groupby(["Feature", "Class"]).size().reset_index(name="Count"))
        chart = (alt.Chart(plot_df).mark_bar().encode(
            x=alt.X("Feature:N", sort="-y", title=pretty(col)),
            y=alt.Y("Count:Q"),
            color=alt.Color("Class:N", scale=alt.Scale(domain=["Edible", "Poisonous"],
                                                      range=["#2e9e5b", "#d6453d"])),
            xOffset="Class:N", tooltip=["Feature", "Class", "Count"]).properties(height=350))
        st.altair_chart(chart, width="stretch")
    with right:
        st.subheader("Most informative features")
        top = importance.head(10).assign(feature=lambda d: d["feature"].map(pretty))
        st.altair_chart(alt.Chart(top).mark_bar(color="#6c5ce7").encode(
            x=alt.X("importance:Q", title="Random Forest importance"),
            y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")]).properties(height=350),
            width="stretch")
    with st.expander("View raw data"):
        st.dataframe(df, width="stretch", height=300)

# ---------- TAB 2: model comparison ----------
with tab_models:
    st.subheader("Test-set performance (80/20 stratified split)")
    st.caption("“Poisonous” is the positive class, so **Recall** = the share of poisonous mushrooms correctly caught.")
    pct_cols = ["Accuracy", "Precision", "Recall (poisonous)", "F1-score", "ROC-AUC", "CV Accuracy (5-fold)"]
    st.dataframe(
        results.style.format({**{c: "{:.2%}" for c in pct_cols}, "CV Std": "{:.4f}", "Train time (s)": "{:.3f}"})
               .background_gradient(subset=["Accuracy"], cmap="Greens"),
        width="stretch", hide_index=True)

    metric = st.selectbox("Compare models by", pct_cols, index=0)
    bar = alt.Chart(results).mark_bar().encode(
        x=alt.X("Model:N", sort="-y", axis=alt.Axis(labelAngle=-30)),
        y=alt.Y(f"{metric}:Q", scale=alt.Scale(domain=[max(0, results[metric].min() - 0.05), 1.0])),
        color=alt.Color("Model:N", legend=None),
        tooltip=["Model", alt.Tooltip(f"{metric}:Q", format=".2%")]).properties(height=350)
    st.altair_chart(bar, width="stretch")

    st.subheader("Confusion matrix")
    model_name = st.selectbox("Model", results["Model"].tolist(), key="cm_model")
    cm = meta["confusion"][model_name]
    cm_df = pd.DataFrame([
        {"Actual": "Edible", "Predicted": "Edible", "Count": cm["TN"]},
        {"Actual": "Edible", "Predicted": "Poisonous", "Count": cm["FP"]},
        {"Actual": "Poisonous", "Predicted": "Edible", "Count": cm["FN"]},
        {"Actual": "Poisonous", "Predicted": "Poisonous", "Count": cm["TP"]}])
    base = alt.Chart(cm_df).encode(x=alt.X("Predicted:N", sort=["Edible", "Poisonous"]),
                                   y=alt.Y("Actual:N", sort=["Edible", "Poisonous"]))
    heat = base.mark_rect().encode(color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), legend=None))
    text = base.mark_text(fontSize=22, fontWeight="bold").encode(
        text="Count:Q", color=alt.condition(alt.datum.Count > cm_df["Count"].max() / 2,
                                            alt.value("white"), alt.value("black")))
    st.altair_chart((heat + text).properties(width=380, height=320))
    if cm["FN"] > 0:
        st.error(f"{cm['FN']} poisonous mushroom(s) were wrongly labelled **edible** — the most dangerous kind of error.")
    else:
        st.success("No poisonous mushrooms were misclassified as edible.")

    st.info(
        "**Observation:** this dataset is almost perfectly separable (e.g. odor and spore-print colour alone "
        "are very strong signals), so most models reach ~100%. Naive Bayes is the exception because it "
        "assumes features are independent, which is not true for mushroom traits.")

# ---------- TAB 3: predict ----------
with tab_predict:
    st.subheader("Describe a mushroom")
    chosen_model = st.selectbox("Algorithm", results["Model"].tolist(), key="pred_model")
    cols_ui = st.columns(3)
    sample = {}
    for i, col in enumerate(meta["columns"]):
        label, mapping = FEATURES[col]
        observed = sorted(df[col].unique())  # only offer values the model has seen
        with cols_ui[i % 3]:
            sample[col] = st.selectbox(
                label, observed, format_func=lambda c, m=mapping: f"{m.get(c, c)} ({c})", key=f"in_{col}")

    if st.button("Classify mushroom", type="primary"):
        model = load_model(meta["model_files"][chosen_model])
        X_new = pd.DataFrame([sample])[meta["columns"]]
        p_poison = float(model.predict_proba(X_new)[0, 1])
        if p_poison >= 0.5:
            st.error(f"### ☠️ Predicted: POISONOUS  \nConfidence: {p_poison:.1%}")
        else:
            st.success(f"### ✅ Predicted: EDIBLE  \nConfidence: {1 - p_poison:.1%}")
        st.progress(p_poison, text=f"Probability poisonous: {p_poison:.1%}")

        with st.expander("What does every model say?"):
            rows = []
            for name, fname in meta["model_files"].items():
                p = float(load_model(fname).predict_proba(X_new)[0, 1])
                rows.append({"Model": name, "Prediction": "Poisonous" if p >= 0.5 else "Edible",
                             "P(poisonous)": f"{p:.1%}"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
