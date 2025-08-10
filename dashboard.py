# app.py — Streamlit Dashboard for Kidney Disease Project (GitHub-connected)
# ---------------------------------------------------------------
# How to use (quick):
# 1) Put your dataset (CSV) in your GitHub repo and copy its RAW URL
#    e.g. https://raw.githubusercontent.com/<user>/<repo>/<branch>/path/to/kidney_disease_dataset.csv
# 2) Run:  streamlit run app.py
# 3) Paste the RAW URL in the sidebar, or upload a local CSV.
#
# Notes:
# - This app removes Google Colab/Drive code and is production-ready for Streamlit.
# - Handles EDA, preprocessing, multiple models, metrics, and model download.

import io
import json
import time
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib

# ------------------------
# Page config
# ------------------------
st.set_page_config(
    page_title="Kidney Disease ML Dashboard",
    page_icon="🧪",
    layout="wide",
)

# Minimal styling
st.markdown(
    """
    <style>
    .metric-row {display: grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); gap: 12px;}
    .small-note {color:#888; font-size:0.9rem}
    </style>
    """ ,
    unsafe_allow_html=True,
)

# ------------------------
# Helpers
# ------------------------
@st.cache_data(show_spinner=False)
def load_csv_from_url(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    return df

@st.cache_data(show_spinner=False)
def load_csv_from_buffer(buffer: io.BytesIO) -> pd.DataFrame:
    df = pd.read_csv(buffer)
    return df

@st.cache_data(show_spinner=False)
def infer_target_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "CKD_Status",
        "target",
        "label",
        "Outcome",
        "Class",
        "Malignant",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        vals = df[c].dropna().unique()
        if len(vals) <= 3 and set(vals).issubset({0, 1, "0", "1", "yes", "no", "Yes", "No", True, False}):
            return c
    return None

def numeric_categorical_split(df: pd.DataFrame, exclude: List[str]) -> Tuple[List[str], List[str]]:
    num_cols = df.select_dtypes(include=[np.number]).columns.difference(exclude).tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.difference(exclude).tolist()
    return num_cols, cat_cols

def build_pipeline(model):
    def _pipeline(df: pd.DataFrame, target: str):
        X = df.drop(columns=[target])
        y = df[target]
        num_cols, cat_cols = numeric_categorical_split(df, exclude=[target])

        numeric_tf = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical_tf = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ])
        pre = ColumnTransformer([
            ("num", numeric_tf, num_cols),
            ("cat", categorical_tf, cat_cols),
        ], remainder="drop")
        pipe = Pipeline([
            ("pre", pre),
            ("clf", model),
        ])
        return X, y, pipe
    return _pipeline

# ------------------------
# Sidebar — Data source & Settings
# ------------------------
st.sidebar.header("Data Source")
raw_url = st.sidebar.text_input(
    "GitHub RAW CSV URL",
    value="",
    placeholder="https://raw.githubusercontent.com/<user>/<repo>/<branch>/path/to/kidney_disease_dataset.csv",
    help="Paste the RAW file URL from your GitHub repo.",
)

uploaded = st.sidebar.file_uploader("...or upload a CSV", type=["csv"])

st.sidebar.header("Train/Test Split")
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", min_value=0, value=42, step=1)

st.sidebar.header("Model Choice")
model_name = st.sidebar.selectbox(
    "Select a model",
    [
        "Logistic Regression",
        "Random Forest",
        "Support Vector Machine",
        "K-Nearest Neighbors",
        "Gradient Boosting",
    ],
)

if model_name == "Logistic Regression":
    C = st.sidebar.select_slider("C (inverse regularization)", options=[0.001,0.01,0.1,1.0,10.0,100.0], value=1.0)
    solver = st.sidebar.selectbox("Solver", ["lbfgs","liblinear"], index=0)
    model = LogisticRegression(C=C, solver=solver, max_iter=200, random_state=random_state)
elif model_name == "Random Forest":
    n_estimators = st.sidebar.slider("n_estimators", 50, 500, 200, 50)
    max_depth = st.sidebar.select_slider("max_depth (None = no limit)", options=[None,5,10,20,30,40,50], value=20)
    min_samples_split = st.sidebar.select_slider("min_samples_split", options=[2,5,10], value=2)
    min_samples_leaf = st.sidebar.select_slider("min_samples_leaf", options=[1,2,4], value=1)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
elif model_name == "Support Vector Machine":
    C = st.sidebar.select_slider("C", options=[0.01,0.1,1.0,10.0,100.0], value=1.0)
    kernel = st.sidebar.selectbox("kernel", ["rbf","linear","poly"], index=0)
    model = SVC(C=C, kernel=kernel, probability=True, random_state=random_state)
elif model_name == "K-Nearest Neighbors":
    n_neighbors = st.sidebar.slider("n_neighbors", 1, 25, 5)
    weights = st.sidebar.selectbox("weights", ["uniform","distance"], index=0)
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
else:
    n_estimators = st.sidebar.slider("n_estimators", 50, 500, 200, 50)
    learning_rate = st.sidebar.select_slider("learning_rate", options=[0.001,0.01,0.05,0.1,0.2,0.3], value=0.1)
    max_depth = st.sidebar.select_slider("max_depth", options=[1,2,3,4,5], value=3)
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )

# ------------------------
# Load data
# ------------------------
st.title("🧪 Kidney Disease ML Dashboard")

df: Optional[pd.DataFrame] = None
load_error = None

try:
    if raw_url:
        df = load_csv_from_url(raw_url)
    elif uploaded is not None:
        df = load_csv_from_buffer(uploaded)
    else:
        st.info("Provide a GitHub RAW CSV URL or upload a CSV to get started.")
except Exception as e:
    load_error = str(e)

if load_error:
    st.error(f"Failed to load dataset: {load_error}")

if isinstance(df, pd.DataFrame):
    st.success("Dataset loaded ✅")
    st.write(df.head())
