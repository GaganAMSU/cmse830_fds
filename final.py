# final_app_with_synthetic.py
"""
Enhanced Streamlit app with two synthetic datasets (MCAR missingness) + sklearn diabetes.
Save as final_app_with_synthetic.py and run: streamlit run final_app_with_synthetic.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import json
import base64
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from functools import partial
import pickle

# Optional libs
try:
    import missingno as msno
    HAS_MISSINGNO = True
except Exception:
    HAS_MISSINGNO = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# ----------------------------
# Config / UI styles
# ----------------------------
st.set_page_config(page_title="Final Project — Enhanced (with Synthetic MCAR datasets)", layout="wide", initial_sidebar_state="expanded")
st.markdown("<style>.title{font-size:26px;font-weight:700}</style>", unsafe_allow_html=True)
st.title("Final Project — Enhanced App (sklearn + 2 synthetic MCAR datasets)")

# ----------------------------
# Sidebar: controls & inputs
# ----------------------------
st.sidebar.header("Dataset controls")
use_builtin = st.sidebar.checkbox("Include sklearn diabetes (builtin)", value=True)

# Synthetic dataset controls (MCAR fractions adjustable)
st.sidebar.markdown("### Synthetic dataset 1 (MCAR)")
mcar1 = st.sidebar.slider("MCAR fraction for synthetic_1 (%)", 0, 60, 10)
nrows1 = st.sidebar.number_input("Rows synthetic_1", min_value=100, max_value=10000, value=500, step=50)

st.sidebar.markdown("### Synthetic dataset 2 (MCAR)")
mcar2 = st.sidebar.slider("MCAR fraction for synthetic_2 (%)", 0, 60, 20)
nrows2 = st.sidebar.number_input("Rows synthetic_2", min_value=100, max_value=10000, value=600, step=50)

st.sidebar.markdown("---")
upload_file = st.sidebar.file_uploader("Upload CSV (optional)", type=['csv'])
external_url = st.sidebar.text_input("External CSV URL (optional)", "")
random_seed = st.sidebar.slider("Random seed", 0, 9999, 42)
test_size = st.sidebar.slider("Test size (%)", 5, 50, 20)
target_column = st.sidebar.text_input("Target column name (if known)", value="target")
st.sidebar.caption("MCAR = Missing Completely At Random (randomly removes values)")

# ----------------------------
# Utility / caching functions
# ----------------------------
@st.cache_data
def load_builtin_diabetes():
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df['source'] = 'sklearn_diabetes'
    # normalize column names to lowercase without spaces to ease integration
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    return df

def _apply_mcar(df: pd.DataFrame, fraction: float, random_state: int = 42):
    """Apply MCAR missingness by randomly selecting fraction of entries (cell-wise) to set NaN."""
    if fraction <= 0:
        return df.copy()
    np.random.seed(random_state)
    df = df.copy()
    # Determine number of total cells
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols
    # number of cells to blank
    n_missing = int(total_cells * fraction)
    if n_missing == 0:
        return df
    # pick random cell indices (row_idx, col_idx)
    # flatten index selection
    choices = np.random.choice(total_cells, size=n_missing, replace=False)
    rows = choices // n_cols
    cols = choices % n_cols
    # set selected cells to NaN (but avoid setting 'source' column to NaN)
    for r, c in zip(rows, cols):
        col_name = df.columns[c]
        if col_name == 'source':
            continue
        df.iat[r, c] = np.nan
    return df

@st.cache_data
def generate_synthetic_1(n_rows: int = 500, mcar_pct: int = 10, random_state: int = 42):
    """
    Synthetic dataset 1:
      - 6 numeric features with varying distributions
      - 1 binary categorical feature
      - continuous target constructed from features + noise
      - MCAR missingness applied at mcar_pct %
    """
    rng = np.random.RandomState(random_state)
    age = rng.normal(50, 12, size=n_rows).clip(18, 90)               # age-like
    bmi = rng.normal(28, 6, size=n_rows).clip(15, 60)               # BMI-like
    systolic = rng.normal(130, 15, size=n_rows).clip(80, 220)
    glucose = rng.exponential(1/0.02, size=n_rows) * 0.5 + 80       # positive skew
    cholesterol = rng.normal(200, 30, size=n_rows)
    smoking = rng.binomial(1, 0.25, size=n_rows).astype(int)
    # target: linear combination + interaction + noise
    target = (0.3*bmi + 0.2*age + 0.4*glucose + 0.1*smoking*10 - 0.15*systolic) + rng.normal(0, 10, size=n_rows)
    df = pd.DataFrame({
        'age': age,
        'bmi': bmi,
        'systolic_bp': systolic,
        'glucose': glucose,
        'cholesterol': cholesterol,
        'smoker': smoking,
        'target': target
    })
    df['source'] = 'synthetic_1'
    # normalize column names
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    # apply MCAR
    df_mcar = _apply_mcar(df, fraction=mcar_pct/100.0, random_state=random_state)
    return df_mcar

@st.cache_data
def generate_synthetic_2(n_rows: int = 600, mcar_pct: int = 20, random_state: int = 123):
    """
    Synthetic dataset 2:
      - 8 numeric features (some correlated), one categorical with 3 levels
      - target non-linear (includes squared term and noise)
      - MCAR missingness applied at mcar_pct %
    """
    rng = np.random.RandomState(random_state)
    feat_a = rng.normal(0, 1, size=n_rows)
    feat_b = feat_a * 0.6 + rng.normal(0, 0.8, size=n_rows)  # correlated with feat_a
    feat_c = rng.uniform(0, 100, size=n_rows)
    feat_d = rng.normal(5, 2, size=n_rows)
    feat_e = rng.poisson(2, size=n_rows)
    cat = rng.choice(['A', 'B', 'C'], size=n_rows, p=[0.5, 0.3, 0.2])
    feat_f = rng.normal(50, 10, size=n_rows)
    feat_g = rng.exponential(1/0.1, size=n_rows) * 0.1
    # target: non-linear combination
    target = 3*feat_a - 2*(feat_b**2) + 0.05*feat_c + 1.5*(feat_d) + rng.normal(0, 5, size=n_rows)
    df = pd.DataFrame({
        'feat_a': feat_a,
        'feat_b': feat_b,
        'feat_c': feat_c,
        'feat_d': feat_d,
        'feat_e': feat_e,
        'category': cat,
        'feat_f': feat_f,
        'feat_g': feat_g,
        'target': target
    })
    df['source'] = 'synthetic_2'
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    df_mcar = _apply_mcar(df, fraction=mcar_pct/100.0, random_state=random_state)
    return df_mcar

@st.cache_data
def load_csv_from_buffer(buffer):
    try:
        df = pd.read_csv(buffer)
    except Exception:
        buffer.seek(0)
        df = pd.read_csv(buffer, encoding='latin1', error_bad_lines=False)
    df['source'] = df.get('source', 'user_upload')
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    return df

@st.cache_data
def load_csv_from_url(url):
    df = pd.read_csv(url)
    df['source'] = df.get('source', 'external_url')
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    return df

# ----------------------------
# Build combined dataset from sources
# ----------------------------
sources_used = []
df_list = []

if use_builtin:
    try:
        df_builtin = load_builtin_diabetes()
        df_list.append(df_builtin)
        sources_used.append('sklearn_diabetes')
    except Exception as e:
        st.error(f"Failed to load builtin diabetes dataset: {e}")

# Generate synthetic datasets with MCAR using sidebar parameters
try:
    df_synth1 = generate_synthetic_1(n_rows=nrows1, mcar_pct=mcar1, random_state=random_seed)
    df_list.append(df_synth1)
    sources_used.append('synthetic_1')
except Exception as e:
    st.error(f"Failed to generate synthetic_1: {e}")

try:
    # use a different random seed for synthetic_2 by default to diversify
    df_synth2 = generate_synthetic_2(n_rows=nrows2, mcar_pct=mcar2, random_state=(random_seed+101))
    df_list.append(df_synth2)
    sources_used.append('synthetic_2')
except Exception as e:
    st.error(f"Failed to generate synthetic_2: {e}")

# optional user-provided sources
if upload_file is not None:
    try:
        df_up = load_csv_from_buffer(upload_file)
        df_list.append(df_up)
        sources_used.append('user_upload')
    except Exception as e:
        st.error(f"Failed to load uploaded CSV: {e}")

if external_url.strip() != "":
    try:
        df_url = load_csv_from_url(external_url.strip())
        df_list.append(df_url)
        sources_used.append('external_url')
    except Exception as e:
        st.error(f"Failed to load CSV from URL: {e}")

if len(df_list) == 0:
    st.warning("No datasets available. Enable builtin or generate synthetic.")
    combined = pd.DataFrame()
else:
    # Normalize columns and concatenate (outer)
    def normalize_columns(df):
        df = df.copy()
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns.astype(str)]
        return df
    df_list_norm = [normalize_columns(d) for d in df_list]
    combined = pd.concat(df_list_norm, axis=0, ignore_index=True, sort=False)
    combined = combined.reset_index(drop=True)

# ----------------------------
# Simple helper utilities for the UI
# ----------------------------
def dataframe_summary(df):
    return {
        'rows': df.shape[0],
        'cols': df.shape[1],
        'missing_values': int(df.isnull().sum().sum()),
        'columns': list(df.columns)
    }

def download_link(df: pd.DataFrame, file_name: str):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}">Download {file_name}</a>'
    return href

def show_numeric_summary(df, cols):
    desc = df[cols].describe().T
    desc['missing'] = df[cols].isnull().sum()
    st.dataframe(desc)

def add_interaction_terms(df, pairs):
    for a, b in pairs:
        name = f"{a}_x_{b}"
        df[name] = df[a] * df[b]
    return df

# ----------------------------
# Page navigation
# ----------------------------
st.sidebar.markdown("---")
page = st.sidebar.radio("Page", [
    "Home",
    "Data Sources & Missingness",
    "EDA & Visualizations",
    "Feature Engineering",
    "Modeling & Evaluation",
    "Documentation & Exports"
])

# ----------------------------
# Home
# ----------------------------
if page == "Home":
    st.markdown("## Overview")
    st.markdown("""
    This enhanced app now includes two synthetic datasets with **MCAR** missingness (user-configurable fraction).
    The three datasets together (builtin + 2 synthetic) satisfy the rubric requirement of having multiple distinct
    data sources to demonstrate integration, missingness handling, advanced EDA, feature engineering and modeling.
    """)
    st.write("Sources currently loaded:", ", ".join(sources_used))
    st.info("Adjust MCAR % for synthetic datasets from the sidebar and re-run (Streamlit will cache results).")

# ----------------------------
# Data Sources & Missingness
# ----------------------------
if page == "Data Sources & Missingness":
    st.subheader("Data Sources Summary")
    if len(df_list) == 0:
        st.info("No data sources loaded.")
    else:
        st.markdown("**Individual source previews**")
        for i, df in enumerate(df_list_norm):
            st.write(f"**Source {i+1}** — name: {df.get('source', 'unknown')}, shape: {df.shape}")
            st.write(dataframe_summary(df))
            st.dataframe(df.head(5))

        st.markdown("### Combined (after normalization/integration)")
        st.write(f"Rows: {combined.shape[0]} — Columns: {combined.shape[1]}")
        st.dataframe(combined.head(6))

        st.markdown("### Missingness overview (per column)")
        missing_count = combined.isnull().sum().sort_values(ascending=False)
        missing_pct = (combined.isnull().mean() * 100).sort_values(ascending=False).round(2)
        missing_df = pd.concat([missing_count, missing_pct], axis=1)
        missing_df.columns = ['missing_count', 'missing_percent']
        st.dataframe(missing_df[missing_df['missing_count'] > 0])

        st.markdown("### Missingness visualizations")
        if HAS_MISSINGNO:
            st.write("Missingness matrix (missingno)")
            fig = msno.matrix(combined, figsize=(10, 4))
            st.pyplot(fig.figure)
            st.write("Missingness bar chart (missingno)")
            fig2 = msno.bar(combined, figsize=(10, 4))
            st.pyplot(fig2.figure)
        else:
            st.write("Fallback missingness heatmap")
            fig, ax = plt.subplots(figsize=(12, 4))
            sns.heatmap(combined.isnull(), cbar=False, yticklabels=False, cmap='viridis', ax=ax)
            ax.set_title("Missingness (True = missing)")
            st.pyplot(fig)

        st.markdown("### Row-level missingness distribution")
        fig, ax = plt.subplots()
        combined.isnull().sum(axis=1).hist(bins=30, ax=ax)
        ax.set_xlabel("Number of missing columns per row")
        st.pyplot(fig)

# ----------------------------
# EDA & Visualizations
# ----------------------------
if page == "EDA & Visualizations":
    st.subheader("Exploratory Data Analysis")
    if combined.empty:
        st.info("Load datasets via the sidebar.")
    else:
        numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = combined.select_dtypes(include=['object', 'category']).columns.tolist()

        st.markdown("### Numeric summary")
        show_numeric_summary(combined, numeric_cols)

        st.markdown("### Visualization selector")
        viz_choice = st.selectbox("Visualization", ["Scatter", "Pairplot", "Violin", "Boxplot", "Correlation Heatmap", "PCA Biplot"])
        x = st.selectbox("X", [None] + numeric_cols)
        y = st.selectbox("Y", [None] + numeric_cols)
        hue = st.selectbox("Hue", [None] + categorical_cols + numeric_cols)

        if viz_choice == "Scatter":
            if x and y:
                fig, ax = plt.subplots()
                sns.scatterplot(data=combined, x=x, y=y, hue=hue, ax=ax)
                st.pyplot(fig)
            else:
                st.info("Select X and Y for scatterplot.")

        if viz_choice == "Pairplot":
            defaults = numeric_cols[:5] if len(numeric_cols) >= 2 else numeric_cols
            sel = st.multiselect("Select features (min 2)", defaults, default=defaults)
            if len(sel) >= 2:
                samp = combined[sel].dropna().sample(min(400, max(100, len(combined))), random_state=random_seed)
                pair_fig = sns.pairplot(samp, diag_kind='kde')
                st.pyplot(pair_fig)
            else:
                st.info("Pick at least two numeric features.")

        if viz_choice == "Violin":
            if categorical_cols and numeric_cols:
                cat = st.selectbox("Categorical", [None] + categorical_cols)
                val = st.selectbox("Value", [None] + numeric_cols)
                if cat and val:
                    fig, ax = plt.subplots(figsize=(10,5))
                    sns.violinplot(x=cat, y=val, data=combined, ax=ax)
                    st.pyplot(fig)
                else:
                    st.info("Choose categorical and numeric columns.")
            else:
                st.info("Need at least one categorical and one numeric column.")

        if viz_choice == "Boxplot":
            if categorical_cols and numeric_cols:
                cat = st.selectbox("Categorical (boxplot)", [None] + categorical_cols)
                val = st.selectbox("Numeric (boxplot)", [None] + numeric_cols)
                if cat and val:
                    fig, ax = plt.subplots()
                    sns.boxplot(x=cat, y=val, data=combined, ax=ax)
                    st.pyplot(fig)
                else:
                    st.info("Choose categorical and numeric columns.")
            else:
                st.info("Need at least one categorical and one numeric column.")

        if viz_choice == "Correlation Heatmap":
            if len(numeric_cols) >= 2:
                fig, ax = plt.subplots(figsize=(10,8))
                sns.heatmap(combined[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                st.pyplot(fig)
            else:
                st.info("Need numeric columns for correlation heatmap.")

        if viz_choice == "PCA Biplot":
            if len(numeric_cols) >= 2:
                n_comp = st.slider("PCA components", 2, min(10, max(2, len(numeric_cols))), 2)
                df_pca = combined[numeric_cols].dropna()
                scaler = StandardScaler()
                Xs = scaler.fit_transform(df_pca)
                pca = PCA(n_components=n_comp, random_state=random_seed)
                pcs = pca.fit_transform(Xs)
                fig, ax = plt.subplots(figsize=(8,6))
                ax.scatter(pcs[:,0], pcs[:,1], alpha=0.6, s=20)
                for i, col in enumerate(df_pca.columns):
                    ax.arrow(0,0, pca.components_[0,i]*5, pca.components_[1,i]*5, color='r', alpha=0.6)
                    ax.text(pca.components_[0,i]*5.2, pca.components_[1,i]*5.2, col, color='r')
                ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("PCA Biplot")
                st.pyplot(fig)
            else:
                st.info("Need at least 2 numeric features for PCA.")

# ----------------------------
# Feature Engineering
# ----------------------------
if page == "Feature Engineering":
    st.subheader("Feature Engineering")
    if combined.empty:
        st.info("Load data first.")
    else:
        numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
        st.write("Numeric cols:", numeric_cols)
        if st.button("Preview Standard scaling"):
            tmp = combined.copy()
            tmp[numeric_cols] = StandardScaler().fit_transform(tmp[numeric_cols].fillna(0))
            st.dataframe(tmp[numeric_cols].head())

        poly_degree = st.selectbox("Polynomial degree (0 = none)", [0,2,3], index=0)
        if poly_degree > 0:
            selected_for_poly = st.multiselect("Select features to poly-expand", numeric_cols, default=numeric_cols[:3])
            if st.button("Show polynomial preview"):
                pf = PolynomialFeatures(degree=poly_degree, include_bias=False)
                arr = pf.fit_transform(combined[selected_for_poly].fillna(0))
                cols = pf.get_feature_names_out(selected_for_poly)
                st.dataframe(pd.DataFrame(arr, columns=cols).head())

        st.markdown("### Interaction terms")
        pair_input = st.text_input("Interaction pairs (col1,col2 per line)", "")
        if st.button("Preview interactions"):
            pairs = []
            for line in pair_input.splitlines():
                parts = [p.strip() for p in line.split(",") if p.strip()!='']
                if len(parts)==2:
                    pairs.append((parts[0], parts[1]))
            if pairs:
                tmp = add_interaction_terms(combined.copy(), pairs)
                st.dataframe(tmp[[f"{a}_x_{b}" for a,b in pairs]].head())
            else:
                st.info("Provide interaction pairs in correct format.")

# ----------------------------
# Modeling & Evaluation
# ----------------------------
if page == "Modeling & Evaluation":
    st.subheader("Modeling & Evaluation")
    if combined.empty:
        st.info("Load the datasets first.")
    else:
        cols_all = combined.columns.tolist()
        selected_target = st.text_input("Target column name", value=(target_column if target_column in cols_all else "target"))
        if selected_target == "":
            st.warning("Specify target column.")
        elif selected_target not in cols_all:
            st.error(f"Target '{selected_target}' not in columns. Available: {cols_all[:10]} ...")
        else:
            df_model = combined.copy()
            df_model = df_model.dropna(subset=[selected_target])
            possible_features = [c for c in df_model.columns if c != selected_target and c != 'source']
            feature_selection = st.multiselect("Select features (leave empty = all numeric)", possible_features,
                                               default=[c for c in possible_features if c in df_model.select_dtypes(include=[np.number]).columns][:8])
            if len(feature_selection) == 0:
                feature_selection = [c for c in df_model.select_dtypes(include=[np.number]).columns if c != selected_target]

            X = df_model[feature_selection].copy()
            y = df_model[selected_target].copy()

            # simple imputation for modeling
            for c in X.columns:
                if X[c].dtype.kind in 'biufc':
                    X[c] = X[c].fillna(X[c].mean())
                else:
                    X[c] = X[c].fillna(X[c].mode().iloc[0] if not X[c].mode().empty else "missing")
            X = pd.get_dummies(X, drop_first=True)

            test_frac = test_size / 100.0
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state=random_seed)

            model_choice = st.selectbox("Model", ["LinearRegression", "Ridge", "Lasso", "RandomForest", "GradientBoosting"])
            use_scaler = st.checkbox("Scale numeric features", value=True)
            if use_scaler:
                scaler = StandardScaler()
                X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
                X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

            if model_choice == "LinearRegression":
                model = LinearRegression()
                param_grid = {'fit_intercept':[True,False]}
            elif model_choice == "Ridge":
                model = Ridge(random_state=random_seed)
                param_grid = {'alpha':[0.01,0.1,1.0,10.0]}
            elif model_choice == "Lasso":
                model = Lasso(random_state=random_seed, max_iter=10000)
                param_grid = {'alpha':[0.001,0.01,0.1,1.0]}
            elif model_choice == "RandomForest":
                model = RandomForestRegressor(random_state=random_seed, n_jobs=-1)
                param_grid = {'n_estimators':[100,200], 'max_depth':[None,10,20], 'min_samples_split':[2,5]}
            else:
                model = GradientBoostingRegressor(random_state=random_seed)
                param_grid = {'n_estimators':[100,200], 'learning_rate':[0.01,0.1], 'max_depth':[3,5]}

            tune = st.checkbox("Perform RandomizedSearchCV tuning", value=False)
            if tune:
                n_iter = st.number_input("n_iter for RandomizedSearchCV", min_value=5, max_value=200, value=20)

            with st.spinner("Training model..."):
                if tune:
                    search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=n_iter, cv=3, random_state=random_seed, n_jobs=-1, scoring='r2')
                    search.fit(X_train, y_train)
                    model = search.best_estimator_
                    st.write("Tuning completed. Best params:", search.best_params_)
                else:
                    model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            st.metric("R² (test)", round(r2,3))
            st.write(f"RMSE: {rmse:.3f} — MAE: {mae:.3f}")

            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
            st.write("Cross-validated R² (5-fold):", np.round(cv_scores,3).tolist(), " mean:", round(cv_scores.mean(),3))

            st.markdown("### Feature importance / coefficients")
            try:
                if hasattr(model, "feature_importances_"):
                    fi = pd.DataFrame({"feature": X_train.columns, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
                    fig, ax = plt.subplots(figsize=(8,6))
                    sns.barplot(data=fi.head(25), x='importance', y='feature', ax=ax)
                    st.pyplot(fig)
                    st.dataframe(fi.head(30))
                elif hasattr(model, "coef_"):
                    coefs = pd.Series(model.coef_, index=X_train.columns).sort_values(key=abs, ascending=False)
                    st.dataframe(coefs.head(30).to_frame("coefficient"))
                    fig, ax = plt.subplots(figsize=(8,6))
                    coefs.head(25).plot.barh(ax=ax); ax.invert_yaxis(); st.pyplot(fig)
                else:
                    st.info("No direct importance available for this model.")
            except Exception as e:
                st.error(f"Failed to compute feature importance: {e}")

            # SHAP optional
            if HAS_SHAP:
                st.markdown("### SHAP explanation (sample)")
                try:
                    explainer = shap.Explainer(model, X_train)
                    shap_values = explainer(X_test.sample(min(100, len(X_test))))
                    st.pyplot(shap.plots.bar(shap_values, show=False))
                except Exception as e:
                    st.error(f"SHAP failed: {e}")
            else:
                st.info("Install `shap` for advanced interpretability.")

            # Offer model download
            model_bytes = pickle.dumps(model)
            b64 = base64.b64encode(model_bytes).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="trained_model.pkl">Download trained model</a>'
            st.markdown(href, unsafe_allow_html=True)

# ----------------------------
# Documentation & Exports
# ----------------------------
if page == "Documentation & Exports":
    st.subheader("Documentation & Exports")
    st.write("Loaded sources:", ", ".join(sources_used))
    if not combined.empty:
        st.markdown("### Combined dataset preview")
        st.dataframe(combined.head(6))
        st.markdown(download_link(combined, "combined_dataset.csv"), unsafe_allow_html=True)

    st.markdown("### Data dictionary (preview)")
    if not combined.empty:
        dict_preview = []
        for c in combined.columns:
            dict_preview.append({
                "column": c,
                "dtype": str(combined[c].dtype),
                "num_missing": int(combined[c].isnull().sum()),
                "unique_preview": str(list(combined[c].dropna().unique()[:5]))
            })
        st.dataframe(pd.DataFrame(dict_preview))
        if st.button("Download data_dictionary.json"):
            s = json.dumps(dict_preview, indent=2, default=str)
            st.download_button("Download JSON", s, file_name="data_dictionary.json")
    else:
        st.info("Load data to create documentation.")

st.markdown("---")
st.caption("This enhanced app includes two synthetic datasets with MCAR missingness. Adjust MCAR percentages and rows in the sidebar and re-run to regenerate data. Good luck on your project!")
