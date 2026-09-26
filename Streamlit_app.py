from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from breast_cancer_predictor.predict import (  # noqa: E402
    ModelArtifactsNotFoundError,
    default_input_row,
    load_artifacts,
)
from breast_cancer_predictor.data import load_dataset  # noqa: E402
from breast_cancer_predictor.train import train_and_evaluate  # noqa: E402


st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")

st.markdown(
    """
    <style>
        .main-title {font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem;}
        .subtitle {color: #385170; margin-top: 0; margin-bottom: 1.4rem;}
        .metric-card {padding: 0.4rem 0.8rem; border-radius: 10px; background: #f5f7fb;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-title">Breast Cancer Prediction Dashboard</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Malignant vs benign classification from 30 cell-nucleus measurements (Wisconsin dataset).</p>',
    unsafe_allow_html=True,
)


@st.cache_data
def get_dataset() -> tuple[pd.DataFrame, list[str]]:
    features, _ = load_dataset()
    return features.copy(), list(features.columns)


@st.cache_resource
def get_model() -> tuple[object, dict]:
    """Load the saved model, training it first if this is a fresh deployment."""
    try:
        return load_artifacts()
    except ModelArtifactsNotFoundError:
        train_and_evaluate()          # a few seconds: the dataset ships with scikit-learn
        return load_artifacts()


model, metrics = get_model()

features_df, feature_names = get_dataset()
default_row = default_input_row()

left_col, right_col = st.columns([1.35, 1])

with right_col:
    st.subheader("Model Snapshot")
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    m2.metric("F1", f"{metrics['f1']:.3f}")
    m3.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
    st.caption("Metrics are computed on a stratified holdout split.")

with left_col:
    st.subheader("Patient Feature Inputs")
    input_payload: dict[str, float] = {}
    for feature in feature_names:
        series = features_df[feature]
        input_payload[feature] = st.slider(
            label=feature,
            min_value=float(series.min()),
            max_value=float(series.max()),
            value=float(default_row.iloc[0][feature]),
            step=float((series.max() - series.min()) / 200) if series.max() > series.min() else 0.01,
        )

input_df = pd.DataFrame([input_payload])

predict_col, preview_col = st.columns([1, 1])
with predict_col:
    if st.button("Run Prediction", type="primary", use_container_width=True):
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]

        benign_prob = float(probabilities[1])
        malignant_prob = float(probabilities[0])

        if prediction == 1:
            st.success("Prediction: Benign")
        else:
            st.error("Prediction: Malignant")

        st.write(f"Benign probability: {benign_prob:.3f}")
        st.write(f"Malignant probability: {malignant_prob:.3f}")

with preview_col:
    st.subheader("Input Preview")
    st.dataframe(input_df, use_container_width=True)
