import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# --- SETUP ---
st.set_page_config(page_title="Diabetes Progression Analysis", layout="wide")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target
    return df

df = load_data()

# --- SIDEBAR ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Introduction",
    "📊 Data Exploration",
    "📈 Visual Analysis",
    "🤖 Predictive Insights",
    "🧠 Conclusions"
])

# --- INTRO PAGE ---
if page == "🏠 Introduction":
    st.title("🩺 Diabetes Progression Explorer")
    st.markdown("""
    ### Project Goal  
    The goal of this project is to explore how **lifestyle and physiological factors** contribute to diabetes progression.  
    Using the `scikit-learn` diabetes dataset, we aim to identify which predictors—such as **BMI**, **blood pressure**, and **blood chemistry**—are most strongly associated with disease progression one year after baseline measurements.
    
    **Dataset:** 442 samples, 10 numeric predictors + continuous target  
    **Methods:** Correlation, Regression, and Interactive Visualization  
    **Tools:** Streamlit, Scikit-learn, Seaborn, Matplotlib
    """)

    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e4/Diabetes_complications_chart.png", 
              caption="Diabetes and its physiological impact", use_container_width=True)

# --- DATA EXPLORATION PAGE ---
elif page == "📊 Data Exploration":
    st.title("📊 Data Exploration")

    st.write("### First 10 Rows of Data")
    st.dataframe(df.head(10))

    st.write("### Summary Statistics")
    st.write(df.describe())

    st.write("### Missing Values Check")
    st.write(df.isnull().sum())

    st.write("### Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
    st.pyplot(fig)

# --- VISUAL ANALYSIS PAGE ---
elif page == "📈 Visual Analysis":
    st.title("📈 Visual Analysis")

    x_var = st.selectbox("Select X-axis variable", df.columns[:-1])
    y_var = st.selectbox("Select Y-axis variable", df.columns[:-1])
    color_by = st.selectbox("Color points by variable", df.columns[:-1])

    st.write(f"#### Scatterplot of {x_var} vs {y_var}")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x_var, y=y_var, hue=color_by, palette="viridis", ax=ax)
    st.pyplot(fig)

    st.write("### Pairplot of Selected Features")
    selected_features = st.multiselect("Select features for pairplot", df.columns[:-1], default=["bmi", "bp", "s5"])
    if len(selected_features) >= 2:
        fig2 = sns.pairplot(df[selected_features])
        st.pyplot(fig2)
    else:
        st.info("Please select at least two features.")

# --- PREDICTIVE INSIGHTS PAGE ---
elif page == "🤖 Predictive Insights":
    st.title("🤖 Predictive Insights")

    X = df.drop('target', axis=1)
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = model.score(X_test, y_test)

    st.metric(label="Model R² Score", value=round(r2, 3))

    st.write("### Feature Importance (Coefficient Magnitude)")
    coef_df = pd.DataFrame({"Feature": df.columns[:-1], "Coefficient": model.coef_})
    coef_df = coef_df.sort_values(by="Coefficient", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=coef_df, x="Coefficient", y="Feature", palette="coolwarm", ax=ax)
    st.pyplot(fig)

    st.write("### Try Custom Predictions")
    user_inputs = {}
    for col in df.columns[:-1]:
        val = st.slider(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        user_inputs[col] = val

    user_df = pd.DataFrame([user_inputs])
    user_scaled = scaler.transform(user_df)
    prediction = model.predict(user_scaled)[0]

    st.success(f"Predicted disease progression: **{prediction:.2f}**")

# --- CONCLUSIONS PAGE ---
elif page == "🧠 Conclusions":
    st.title("🧠 Key Findings & Conclusions")

    st.markdown("""
    ### Summary of Insights:
    - **BMI** and **blood chemistry measures (s5, s6)** show strong positive correlation with diabetes progression.  
    - **Regression analysis** suggests BMI has the largest coefficient, indicating its strong effect.  
    - The **R² score** demonstrates how much variance in progression is explained by the predictors.  
    - Lifestyle and body composition changes may play a significant role in slowing disease progression.

    ### Next Steps:
    - Include additional datasets (e.g., CDC BRFSS) for a larger sample.  
    - Explore non-linear models (Random Forest, Gradient Boosting).  
    - Integrate external health indicators (activity, diet, glucose level).

    **Thank you!**
    """)

    st.balloons()
