# final_app_green.py
"""
Final Streamlit App (Green + White Theme)
✓ sklearn diabetes
✓ Pima Indians Diabetes dataset
✓ NHANES curated sample
✓ Upload/URL datasets
✓ Missingness, EDA, Feature Engineering, Modeling
✓ Themed UI with green + white styling
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import base64
import json
import pickle

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, cross_validate, learning_curve
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    import missingno as msno
    HAS_MISSINGNO = True
except:
    HAS_MISSINGNO = False

try:
    import shap
    HAS_SHAP = True
except:
    HAS_SHAP = False

# -----------------------------------
# STREAMLIT CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Diabetes Analysis App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------
# CUSTOM GREEN & WHITE THEME
# -----------------------------------
st.markdown("""
<style>

body {
    background-color: #ffffff !important;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #eaf7ea !important;
}

/* Headers */
h1, h2, h3, h4 {
    color: #0b6e0b !important;
    font-weight: 700 !important;
}

/* Buttons */
.stButton>button {
    background-color: #0b6e0b !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 0.4rem 1rem !important;
    border: none !important;
}

.stButton>button:hover {
    background-color: #0f8b0f !important;
}

/* Dataframes rounded */
.dataframe {
    border-radius: 8px !important;
    overflow: hidden !important;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------------
# TITLE
# -----------------------------------
st.title("Diabetes Analysis App")

# -----------------------------------
# SIDEBAR CONTROLS
# -----------------------------------
st.sidebar.header("Data Sources")

use_builtin = st.sidebar.checkbox("Include sklearn diabetes dataset", True)
include_pima = st.sidebar.checkbox("Include Pima Indians Diabetes dataset", True)
include_nhanes = st.sidebar.checkbox("Include NHANES curated sample", True)

uploaded = st.sidebar.file_uploader("Upload your CSV(optional)", type=['csv'])
external_url = st.sidebar.text_input("External CSV URL(optional)")

seed = st.sidebar.number_input("Random Seed", 0, 9999, 42)
test_size = st.sidebar.slider("Test size (%)", 5, 50, 20)
target_column = st.sidebar.text_input("Target column", "target")

# -----------------------------------
# DATA LOADERS
# -----------------------------------
@st.cache_data
def load_builtin():
    d = load_diabetes()
    df = pd.DataFrame(d.data, columns=d.feature_names)
    df["target"] = d.target
    df["source"] = "sklearn_diabetes"
    df.columns = df.columns.str.lower()
    return df

@st.cache_data
def load_pima(url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"):
    df = pd.read_csv(url, header=None)
    df.columns = [
        "pregnancies", "glucose", "blood_pressure", "skin_thickness",
        "insulin", "bmi", "diabetes_pedigree", "age", "outcome"
    ]
    df["target"] = df["outcome"]
    df.drop(columns=["outcome"], inplace=True)
    df["source"] = "pima"
    df.columns = df.columns.str.lower()
    return df

@st.cache_data
def load_nhanes(url="https://raw.githubusercontent.com/statOmics/PSLSData/main/NHANES.csv"):
    df = pd.read_csv(url)
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    def find(kws):
        for k in kws:
            for c in df.columns:
                if k in c:
                    return c
        return None

    mapping = {
        "age": find(["age"]),
        "bmi": find(["bmi", "body_mass", "bmx"]),
        "systolic_bp": find(["bpsys", "systolic", "bpxsy"]),
        "glucose": find(["glucose"]),
        "cholesterol": find(["chol"])
    }

    reduced = pd.DataFrame()
    for new, old in mapping.items():
        reduced[new] = df[old] if old else np.nan

    reduced["source"] = "nhanes"
    return reduced

@st.cache_data
def load_uploaded(buffer):
    df = pd.read_csv(buffer)
    df["source"] = "user_upload"
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df

@st.cache_data
def load_from_url(url):
    df = pd.read_csv(url)
    df["source"] = "external"
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df

# -----------------------------------
# COMBINE DATASETS
# -----------------------------------
df_list = []
sources = []

if use_builtin:
    df_list.append(load_builtin())
    sources.append("sklearn_diabetes")

if include_pima:
    df_list.append(load_pima())
    sources.append("pima")

if include_nhanes:
    df_list.append(load_nhanes())
    sources.append("nhanes")

if uploaded:
    df_list.append(load_uploaded(uploaded))
    sources.append("user_upload")

if external_url.strip():
    df_list.append(load_from_url(external_url.strip()))
    sources.append("external_url")

combined = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

# -----------------------------------
# PAGE NAVIGATION
# -----------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Missingness", "EDA", "Feature Engineering", "Modeling", "Documentation"]
)

# -----------------------------------
# HOME PAGE
# -----------------------------------
if page == "Home":
    st.header("💚 Welcome")
    st.markdown(f"""
### **Datasets Loaded:**  
**{", ".join(sources)}**

This app provides:
- Missingness analysis  
- Exploratory Data Analysis  
- Visualizations  
- Feature Engineering  
- Machine Learning Modeling  
""")

# -----------------------------------
# MISSINGNESS PAGE
# -----------------------------------
if page == "Missingness":
    st.header("🔍 Missingness Analysis")

    if combined.empty:
        st.warning("No data")
    else:
        st.subheader("Dataset Preview")
        st.dataframe(combined.head())

        st.subheader("Column Missingness")
        miss = combined.isnull().sum().to_frame("missing_count")
        miss["missing_percent"] = (combined.isnull().mean() * 100).round(2)
        st.dataframe(miss)

        st.subheader("Missingness Visualization")
        if HAS_MISSINGNO:
            fig = msno.matrix(combined)
            st.pyplot(fig.figure)
        else:
            fig, ax = plt.subplots(figsize=(12, 4))
            sns.heatmap(combined.isnull(), cbar=False, cmap="Greens", yticklabels=False, ax=ax)
            st.pyplot(fig)

# -----------------------------------
# EDA PAGE
# -----------------------------------
if page == "EDA":
    st.header("📊 Exploratory Data Analysis")

    if combined.empty:
        st.warning("No data")
    else:
        numeric = combined.select_dtypes(include=[np.number]).columns.tolist()
        categorical = combined.select_dtypes(include=["object"]).columns.tolist()

        st.subheader("Summary Statistics")
        st.dataframe(combined.describe().T)

        st.subheader("Visualizations")
        opt = st.selectbox("Plot type", ["Scatter", "Heatmap", "Boxplot"])

        if opt == "Scatter":
            x = st.selectbox("X", numeric)
            y = st.selectbox("Y", numeric)
            hue = st.selectbox("Hue", [None] + categorical)
            fig, ax = plt.subplots()
            sns.scatterplot(data=combined, x=x, y=y, hue=hue, ax=ax, palette="Greens")
            st.pyplot(fig)

        if opt == "Heatmap":
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(combined[numeric].corr(), annot=True, cmap="Greens", ax=ax)
            st.pyplot(fig)

        if opt == "Boxplot":
            cat = st.selectbox("Category", categorical)
            num = st.selectbox("Value", numeric)
            fig, ax = plt.subplots()
            sns.boxplot(data=combined, x=cat, y=num, ax=ax, palette="Greens")
            st.pyplot(fig)

# -----------------------------------
# FEATURE ENGINEERING PAGE
# -----------------------------------
if page == "Feature Engineering":
    st.header("🧰 Feature Engineering")

    if combined.empty:
        st.warning("No data")
    else:
        numeric = combined.select_dtypes(include=[np.number]).columns.tolist()

        st.subheader("Standard Scaling")
        if st.button("Preview Scaling"):
            tmp = combined.copy()
            tmp[numeric] = StandardScaler().fit_transform(tmp[numeric].fillna(0))
            st.dataframe(tmp.head())

        st.subheader("Polynomial Features")
        deg = st.slider("Degree", 1, 4, 2)
        selected = st.multiselect("Select features", numeric, default=numeric[:3])

        if st.button("Generate Polynomial Features"):
            pf = PolynomialFeatures(degree=deg, include_bias=False)
            arr = pf.fit_transform(combined[selected].fillna(0))
            cols = pf.get_feature_names_out(selected)
            st.dataframe(pd.DataFrame(arr, columns=cols).head())

# -----------------------------------
# MODELING PAGE
# -----------------------------------
if page == "Modeling":
    st.header("🤖 Advanced Modeling & Model Explainability")

    if combined.empty:
        st.warning("No data loaded.")
    else:
        if target_column not in combined.columns:
            st.error(f"Target '{target_column}' not found.")
        else:
            df_m = combined.dropna(subset=[target_column])
            features = [c for c in df_m.columns if c not in ["source", target_column]]

            selected = st.multiselect("Select Features", features, default=features[:6])

            X = pd.get_dummies(df_m[selected], drop_first=True).fillna(0)
            y = df_m[target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size / 100, random_state=seed
            )

            # Available models
            model_type = st.selectbox("Model", [
                "LinearRegression", "Ridge", "Lasso", "RandomForest", "GradientBoosting"
            ])

            # Choose model
            if model_type == "LinearRegression":
                model = LinearRegression()
            elif model_type == "Ridge":
                model = Ridge(random_state=seed)
            elif model_type == "Lasso":
                model = Lasso(max_iter=5000, random_state=seed)
            elif model_type == "RandomForest":
                model = RandomForestRegressor(random_state=seed)
            else:
                model = GradientBoostingRegressor(random_state=seed)

            # Hyperparameter tuning
            tune = st.checkbox("Enable Hyperparameter Tuning")

            param_grids = {
                "LinearRegression": {
                    "fit_intercept": [True, False],
                    "positive": [False, True]
                },
                "Ridge": {
                    "alpha": [0.01, 0.1, 1.0, 5.0, 10.0],
                    "fit_intercept": [True, False]
                },
                "Lasso": {
                    "alpha": [0.001, 0.01, 0.1, 1.0],
                    "max_iter": [2000, 5000, 8000]
                },
                "RandomForest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10]
                },
                "GradientBoosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [2, 3, 4]
                }
            }

            if tune:
                st.write("🔍 Performing hyperparameter tuning…")
                search = RandomizedSearchCV(
                    model,
                    param_distributions=param_grids[model_type],
                    n_iter=10,
                    cv=3,
                    scoring="r2",
                    random_state=seed,
                    n_jobs=-1
                )
                search.fit(X_train, y_train)
                model = search.best_estimator_
                st.success("🎯 Best Params Found:")
                st.write(search.best_params_)
            else:
                model.fit(X_train, y_train)

            # Predictions
            preds = model.predict(X_test)

            # Metrics
            st.subheader("📊 Performance Metrics")
            r2 = r2_score(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)

            col1, col2, col3 = st.columns(3)
            col1.metric("R² Score", f"{r2:.4f}")
            col2.metric("RMSE", f"{rmse:.4f}")
            col3.metric("MAE", f"{mae:.4f}")

            # ------------- Residual Diagnostics --------------
            st.subheader("📉 Residual Diagnostics")

            residuals = y_test - preds

            fig, ax = plt.subplots()
            sns.scatterplot(x=preds, y=residuals, ax=ax)
            ax.axhline(0, color="red", linestyle="--")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Residuals")
            ax.set_title("Residuals vs Predicted")
            st.pyplot(fig)

            # Residual histogram
            fig, ax = plt.subplots()
            sns.histplot(residuals, kde=True, ax=ax)
            ax.set_title("Residual Distribution")
            st.pyplot(fig)

            # Q-Q plot
            fig = plt.figure()
            sm.qqplot(residuals, line='45', fit=True)
            st.pyplot(fig)

            # ------------- Prediction Error Plot --------------
            st.subheader("📈 Prediction Error Plot")
            fig, ax = plt.subplots()
            sns.scatterplot(x=y_test, y=preds, ax=ax)
            ax.set_xlabel("True Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("True vs Predicted")
            st.pyplot(fig)

            # ------------- Cross-Validation -------------------
            st.subheader("📚 Cross-Validation (5-fold)")

            cv_scores = cross_validate(
                model, X, y, cv=5,
                scoring=("r2", "neg_mean_squared_error", "neg_mean_absolute_error")
            )

            cv_df = pd.DataFrame({
                "Fold": range(1, 6),
                "R²": cv_scores["test_r2"],
                "RMSE": np.sqrt(-cv_scores["test_neg_mean_squared_error"]),
                "MAE": -cv_scores["test_neg_mean_absolute_error"]
            })
            st.dataframe(cv_df)

            # ------------- Learning Curve ---------------------
            st.subheader("📉 Learning Curve")

            train_sizes, train_scores, test_scores = learning_curve(
                model, X, y, cv=5,
                train_sizes=np.linspace(0.1, 1.0, 5),
                scoring="r2"
            )

            fig, ax = plt.subplots()
            ax.plot(train_sizes, train_scores.mean(axis=1), label="Train Score")
            ax.plot(train_sizes, test_scores.mean(axis=1), label="Val Score")
            ax.set_xlabel("Training Size")
            ax.set_ylabel("R² Score")
            ax.legend()
            st.pyplot(fig)

            # ------------- SHAP EXPLAINABILITY ----------------
            st.subheader("🧠 SHAP Explainability")

            try:
                explainer = shap.Explainer(model, X_train)
                shap_values = explainer(X_test)

                st.write("### SHAP Summary Plot")
                fig = shap.plots.beeswarm(shap_values, show=False)
                st.pyplot(bbox_inches="tight")

                st.write("### SHAP Bar Plot")
                fig = shap.plots.bar(shap_values, show=False)
                st.pyplot(bbox_inches="tight")

            except:
                st.info("SHAP not supported for this model type.")

            # ------------- Model Download ---------------------
            st.subheader("⬇️ Download Model")
            b = pickle.dumps(model)
            b64 = base64.b64encode(b).decode()

            st.markdown(
                f'<a href="data:application/octet-stream;base64,{b64}" '
                f'download="model.pkl" style="font-size:18px;color:#0b6e0b;">'
                f'Download model.pkl</a>',
                unsafe_allow_html=True
            )

# -----------------------------------
# DOCUMENTATION PAGE
# -----------------------------------
if page == "Documentation":
    st.header("📄 Documentation & Exports")

    if combined.empty:
        st.warning("No data available to document.")
    
    else:
        # -----------------------------------
        # PREVIEW SECTION
        # -----------------------------------
        st.subheader("🔍 Dataset Preview")
        st.dataframe(combined.head())

        # -----------------------------------
        # DOWNLOAD COMBINED DATASET
        # -----------------------------------
        st.subheader("📥 Download Combined Dataset")
        csv = combined.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()

        st.markdown(
            f'<a href="data:file/csv;base64,{b64}" download="combined_dataset.csv" '
            f'style="font-size:18px;color:#0b6e0b;font-weight:bold;">'
            f'⬇️ Download CSV</a>',
            unsafe_allow_html=True
        )


