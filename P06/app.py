import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="MLDM P06: Model Recommendation Explainer", layout="wide")

st.title("MLDM P06: Model Recommendation Explainer")
st.caption(
    "This app explains *model behaviour* (what the model did and why), not real-world cause and effect."
)

ROOT = Path(__file__).parent


# -----------------------------
# Helpers
# -----------------------------
def find_target_column(df: pd.DataFrame) -> str | None:
    """
    Best-effort guess of target column without asking the user.
    Falls back to last column if unsure.
    """
    candidates = [
        "target", "label", "class", "y", "diagnosis", "outcome", "is_fraud", "default"
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    # If none matched, return last column (common in teaching datasets)
    if len(df.columns) > 0:
        return df.columns[-1]
    return None


def load_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def load_data():
    dataset_path = ROOT / "dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError("dataset.csv not found in the same folder as app.py")

    df = pd.read_csv(dataset_path)

    df_test_idx = load_csv_if_exists(ROOT / "test_idx.csv")
    df_train_idx = load_csv_if_exists(ROOT / "train_idx.csv")
    df_pred = load_csv_if_exists(ROOT / "test_predictions.csv")

    df_fi = load_csv_if_exists(ROOT / "dt_feature_importance.csv")
    df_pi = load_csv_if_exists(ROOT / "permutation_importance.csv")

    versions = None
    vpath = ROOT / "VERSIONS.json"
    if vpath.exists():
        try:
            versions = json.loads(vpath.read_text(encoding="utf-8"))
        except Exception:
            versions = None

    return df, df_train_idx, df_test_idx, df_pred, df_fi, df_pi, versions


@st.cache_resource(show_spinner=False)
def load_model():
    model_path = ROOT / "dt_best_pipeline.joblib"
    if not model_path.exists():
        raise FileNotFoundError("dt_best_pipeline.joblib not found in the same folder as app.py")
    return joblib.load(model_path)


def get_feature_names_from_preprocessor(preprocessor) -> list[str] | None:
    """
    Works for sklearn ColumnTransformer / Pipeline preprocessor that supports get_feature_names_out.
    Returns None if not available.
    """
    try:
        names = preprocessor.get_feature_names_out()
        return [str(n) for n in names]
    except Exception:
        return None


def extract_pipeline_parts(pipeline):
    """
    Tries common names: ("preprocessor", "classifier") or ("preprocessor", "model") etc.
    If not found, falls back to last step as estimator.
    """
    preprocessor = None
    estimator = None

    if hasattr(pipeline, "named_steps"):
        if "preprocessor" in pipeline.named_steps:
            preprocessor = pipeline.named_steps["preprocessor"]

        # common estimator step names
        for k in ["classifier", "model", "regressor", "estimator"]:
            if k in pipeline.named_steps:
                estimator = pipeline.named_steps[k]
                break

        if estimator is None:
            # last step
            try:
                estimator = list(pipeline.named_steps.values())[-1]
            except Exception:
                estimator = None

    return preprocessor, estimator


def decision_tree_rules_for_row(tree, feature_names: list[str], x_row_2d: np.ndarray) -> list[str]:
    """
    Produces a readable list of rules along the decision path for one row.
    x_row_2d must be shape (1, n_features).
    """
    if x_row_2d.ndim != 2 or x_row_2d.shape[0] != 1:
        raise ValueError("x_row_2d must have shape (1, n_features)")

    node_indicator = tree.decision_path(x_row_2d)
    leaf_id = tree.apply(x_row_2d)[0]

    feature = tree.tree_.feature
    threshold = tree.tree_.threshold

    node_index = node_indicator.indices[node_indicator.indptr[0] : node_indicator.indptr[1]]

    rules = []
    for node_id in node_index:
        if node_id == leaf_id:
            continue

        f_id = feature[node_id]
        if f_id < 0:
            continue

        f_name = feature_names[f_id] if f_id < len(feature_names) else f"feature_{f_id}"
        t = threshold[node_id]
        v = x_row_2d[0, f_id]

        # Determine direction: left means <= threshold, right means > threshold
        if v <= t:
            rules.append(f"- {f_name} <= {t:.4g}  (your value: {v:.4g})")
        else:
            rules.append(f"- {f_name} >  {t:.4g}  (your value: {v:.4g})")

    return rules


def plot_importance(df_imp: pd.DataFrame, title: str, top_n: int = 15):
    """
    Accepts a few common importance CSV formats.
    Expected columns often include: feature, importance OR importances_mean, etc.
    """
    if df_imp is None or df_imp.empty:
        st.info(f"No data available for: {title}")
        return

    cols = [c.lower() for c in df_imp.columns]
    df = df_imp.copy()

    # Guess feature column
    feature_col = None
    for c in df.columns:
        if c.lower() in ["feature", "features", "name", "variable"]:
            feature_col = c
            break
    if feature_col is None:
        feature_col = df.columns[0]

    # Guess importance column
    imp_col = None
    for c in df.columns:
        if c.lower() in ["importance", "importances_mean", "mean_importance", "score", "weight"]:
            imp_col = c
            break
    if imp_col is None:
        # fallback: second column if exists
        imp_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    df = df[[feature_col, imp_col]].dropna()
    df = df.sort_values(by=imp_col, ascending=False).head(top_n)

    fig = plt.figure()
    plt.barh(df[feature_col].astype(str)[::-1], df[imp_col][::-1])
    plt.title(title)
    plt.xlabel("Importance (higher means more influence in the model)")
    plt.tight_layout()
    st.pyplot(fig)


# -----------------------------
# Load artefacts
# -----------------------------
with st.spinner("Loading artefacts..."):
    df, df_train_idx, df_test_idx, df_pred, df_fi, df_pi, versions = load_data()
    pipe = load_model()

target_col = find_target_column(df)
if target_col is None:
    st.error("Could not infer a target column from dataset.csv")
    st.stop()

# Build X/y (best effort: assume target_col is label)
X_all = df.drop(columns=[target_col], errors="ignore")
y_all = df[target_col] if target_col in df.columns else None

# Identify test indices
test_indices = None
if df_test_idx is not None and not df_test_idx.empty:
    # common formats: a column called "idx" or the only column
    if "idx" in df_test_idx.columns:
        test_indices = df_test_idx["idx"].astype(int).tolist()
    else:
        test_indices = df_test_idx.iloc[:, 0].astype(int).tolist()

if test_indices is None:
    # fallback: allow picking any row
    test_indices = list(range(len(df)))

# Extract pipeline parts
preprocessor, estimator = extract_pipeline_parts(pipe)
feature_names = None
if preprocessor is not None:
    feature_names = get_feature_names_from_preprocessor(preprocessor)

# If feature names still unknown, create generic names
if feature_names is None:
    # Try to infer number of transformed features by transforming one row
    try:
        x0 = X_all.iloc[[0]]
        X0t = preprocessor.transform(x0) if preprocessor is not None else x0.to_numpy()
        n_feat = X0t.shape[1]
        feature_names = [f"feature_{i}" for i in range(n_feat)]
    except Exception:
        feature_names = [f"feature_{i}" for i in range(999)]  # safe fallback


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Pick a case")
st.sidebar.write("Choose a row index to explain (from test_idx.csv if available).")

chosen_global_idx = st.sidebar.selectbox("Row index", options=test_indices, index=0)

show_raw_row = st.sidebar.checkbox("Show raw input row", value=True)
show_pred_file = st.sidebar.checkbox("Show test_predictions.csv row (if available)", value=False)

st.sidebar.header("Display options")
top_n = st.sidebar.slider("Top N features (global charts)", min_value=5, max_value=30, value=15, step=1)


# -----------------------------
# Prepare selected row
# -----------------------------
x_row = X_all.loc[[chosen_global_idx]].copy()  # keep as DataFrame (1 row)
y_true = None
if y_all is not None and chosen_global_idx in y_all.index:
    y_true = y_all.loc[chosen_global_idx]

# Predict
try:
    pred = pipe.predict(x_row)[0]
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

proba = None
if hasattr(pipe, "predict_proba"):
    try:
        proba = pipe.predict_proba(x_row)[0]
    except Exception:
        proba = None

# Transform for tree explanation
x_row_transformed = None
if preprocessor is not None:
    try:
        x_row_transformed = preprocessor.transform(x_row)
        # ensure numpy 2D
        if not isinstance(x_row_transformed, np.ndarray):
            x_row_transformed = x_row_transformed.toarray() if hasattr(x_row_transformed, "toarray") else np.asarray(x_row_transformed)
    except Exception:
        x_row_transformed = None
else:
    x_row_transformed = x_row.to_numpy()


# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("1) Recommendation (prediction)")
    if y_true is not None:
        st.write(f"**True label (from dataset.csv):** `{y_true}`")

    st.write(f"**Model prediction:** `{pred}`")

    if proba is not None:
        # Display probabilities with class labels if possible
        class_labels = None
        try:
            class_labels = list(pipe.classes_)
        except Exception:
            class_labels = [f"class_{i}" for i in range(len(proba))]

        proba_df = pd.DataFrame({"class": class_labels, "probability": proba}).sort_values("probability", ascending=False)
        st.dataframe(proba_df, use_container_width=True, hide_index=True)
    else:
        st.info("This pipeline does not expose predict_proba, so probabilities are unavailable.")

    if show_raw_row:
        st.subheader("Input features (raw)")
        st.dataframe(x_row, use_container_width=True)

    if show_pred_file and df_pred is not None and not df_pred.empty:
        st.subheader("Recorded test prediction row (from test_predictions.csv)")
        # best effort: match by index if a column exists, else show a slice
        idx_cols = [c for c in df_pred.columns if c.lower() in ["idx", "index", "row_id", "row"]]
        if idx_cols:
            m = df_pred[df_pred[idx_cols[0]].astype(int) == int(chosen_global_idx)]
            if len(m) > 0:
                st.dataframe(m, use_container_width=True, hide_index=True)
            else:
                st.info("No matching row found in test_predictions.csv for this index.")
        else:
            st.dataframe(df_pred.head(10), use_container_width=True)


with right:
    st.subheader("2) Why the model recommended this (local explanation)")

    # Prefer tree rule explanation (very student-friendly for DecisionTree)
    tree = estimator
    if tree is not None and hasattr(tree, "tree_") and x_row_transformed is not None:
        try:
            rules = decision_tree_rules_for_row(tree, feature_names, np.asarray(x_row_transformed))
            if len(rules) == 0:
                st.info("No split rules were found (unexpected for a DecisionTree), or this case went straight to the root leaf.")
            else:
                st.write("**Decision rules used for this case:**")
                st.markdown("\n".join(rules))

            st.caption(
                "Interpretation tip: each rule is one gate the case passed through before landing at a leaf node."
            )
        except Exception as e:
            st.warning(f"Could not build decision rules: {e}")
    else:
        st.info(
            "Local explanation is unavailable because the estimator or transformed features could not be accessed."
        )

    st.subheader("3) What matters overall (global importance)")
    c1, c2 = st.columns(2)

    with c1:
        st.write("**Decision Tree feature importance**")
        plot_importance(df_fi, "Decision Tree feature importance (top)", top_n=top_n)

    with c2:
        st.write("**Permutation importance**")
        plot_importance(df_pi, "Permutation importance (top)", top_n=top_n)


st.divider()
st.subheader("Notes for students")
st.write(
    "- A high global importance means a feature often influences predictions across many cases.\n"
    "- The local rules show why *this one case* ended up with its prediction.\n"
    "- These explanations show model behaviour, not cause and effect in the real world."
)

if versions is not None:
    with st.expander("Environment versions (from VERSIONS.json)"):
        st.json(versions)
