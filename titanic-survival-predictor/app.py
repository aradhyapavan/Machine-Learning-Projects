import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# Optional explainability libraries
try:
    import shap  # type: ignore
    _shap_available = True
except Exception:
    _shap_available = False

try:
    from lime.lime_tabular import LimeTabularExplainer  # type: ignore
    _lime_available = True
except Exception:
    _lime_available = False

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# Check if model files exist
model_path = 'titanic_decision_tree_model.joblib'
features_path = 'feature_names.joblib'

if not os.path.exists(model_path) or not os.path.exists(features_path):
    st.error("Model files not found. Please run the notebook to export the model first.")
    st.stop()

# Load the saved model and feature names
dt_model = joblib.load(model_path)
feature_names = joblib.load(features_path)

# Map technical feature names to friendly, human-readable labels
def to_friendly_feature_name(raw_name: str) -> str:
    try:
        if raw_name == 'Age':
            return 'Age (years)'
        if raw_name == 'Fare':
            return 'Fare (£)'
        if raw_name.startswith('Pclass_'):
            cls = raw_name.split('_', 1)[1]
            label = {'1': '1st Class', '2': '2nd Class', '3': '3rd Class'}.get(cls, cls)
            return f'Pclass: {label}'
        if raw_name.startswith('Sex_'):
            sex = raw_name.split('_', 1)[1]
            sex_label = 'Male' if sex.lower() == 'male' else 'Female'
            return f'Sex: {sex_label}'
        if raw_name.startswith('SibSp_'):
            num = raw_name.split('_', 1)[1]
            return f'Siblings/Spouses: {num}'
        if raw_name.startswith('Parch_'):
            num = raw_name.split('_', 1)[1]
            return f'Parents/Children: {num}'
        if raw_name.startswith('Embarked_'):
            port = raw_name.split('_', 1)[1]
            port_label = {'S': 'Southampton (S)', 'C': 'Cherbourg (C)', 'Q': 'Queenstown (Q)'}.get(port, port)
            return f'Embarked: {port_label}'
        return raw_name
    except Exception:
        return raw_name

friendly_feature_names = [to_friendly_feature_name(n) for n in feature_names]

# Background data for explainability
@st.cache_data(show_spinner=False)
def load_background_data():
    try:
        df = pd.read_csv('data_cleaned.csv')
        if 'Survived' in df.columns:
            X = df.drop(columns=['Survived'])
        else:
            X = df.copy()
        # Reorder to match training features
        X = X[[c for c in feature_names if c in X.columns]]
        return df, X
    except Exception:
        return None, None

bg_df, bg_X = load_background_data()

# Cache explainers so they initialize only once
@st.cache_resource(show_spinner=False)
def get_shap_explainer(_model, _background_frame):
    if not _shap_available or _background_frame is None or len(_background_frame) == 0:
        return None
    try:
        return shap.Explainer(_model, _background_frame, feature_names=friendly_feature_names)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def get_lime_explainer(_background_frame):
    if not _lime_available or _background_frame is None or len(_background_frame) == 0:
        return None
    try:
        return LimeTabularExplainer(
            training_data=_background_frame.values,
            feature_names=friendly_feature_names,
            class_names=['Not Survive', 'Survive'],
            discretize_continuous=True,
            mode='classification'
        )
    except Exception:
        return None

# App title and description
st.title("🚢 Titanic Survival Predictor")
st.markdown("""
This app predicts whether a passenger would have survived the Titanic disaster based on 
their characteristics using a **Decision Tree model**. Enter the passenger details below and click 'Predict Survival'.
""")

# Add model performance metrics
st.info("""
**Model Performance**:
- Training Score: 88.0%
- Validation Score: 81.6%
""")

# Create a function to prepare the input data
def prepare_input_data():
    # Start with zeros for all features
    input_data = {feature: 0 for feature in feature_names}
    
    # Set values for numeric features
    input_data['Age'] = age
    input_data['Fare'] = fare
    
    # Set one-hot encoded features
    # For Pclass
    pclass_mapping = {"1st Class": "Pclass_1", "2nd Class": "Pclass_2", "3rd Class": "Pclass_3"}
    if pclass_mapping[pclass] in feature_names:
        input_data[pclass_mapping[pclass]] = 1
    
    # For Sex
    sex_feature = f"Sex_{'male' if sex == 'Male' else 'female'}"
    if sex_feature in feature_names:
        input_data[sex_feature] = 1
    
    # For SibSp
    sibsp_feature = f"SibSp_{sibsp}"
    if sibsp_feature in feature_names:
        input_data[sibsp_feature] = 1
    
    # For Parch
    parch_feature = f"Parch_{parch}"
    if parch_feature in feature_names:
        input_data[parch_feature] = 1
    
    # For Embarked
    embarked_mapping = {
        "Southampton (S)": "Embarked_S", 
        "Cherbourg (C)": "Embarked_C", 
        "Queenstown (Q)": "Embarked_Q"
    }
    if embarked_mapping[embarked] in feature_names:
        input_data[embarked_mapping[embarked]] = 1
        
    # Convert to DataFrame with features in the right order
    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_names]  # Ensure columns are in the same order
    
    return input_df

# Updated README content
readme_content = """
# Titanic Survivor Prediction: Predicting Survival on the Titanic Ship

This repository houses the code and analysis for a Machine Learning project aimed at predicting the survival of passengers aboard the RMS Titanic. Explore the tragic incident through the lens of data science and build a model that sheds light on factors influencing survival rates.

## Problem Statement:

On April 15, 1912, the maiden voyage of the supposedly "unsinkable" Titanic ended in tragedy after striking an iceberg. Out of 2224 passengers and crew onboard, 1502 perished, leaving a lasting mark on maritime history. While chance played a role, certain demographics demonstrably had higher survival rates. This project aims to uncover these patterns by building a predictive model for passenger survival.

## Data Description:

The dataset utilized in this project includes information on each passenger, including:

- **Pclass**: Ticket class (1st, 2nd, 3rd)
- **Sex**: Gender (male, female)
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Passenger fare
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation (Cherbourg, Queenstown, Southampton)
- **Survival**: Survival status (0: No, 1: Yes)

## Project Highlights:

- **Data Exploration and Cleaning**: Analyze passenger demographics, identify missing values, and prepare the data for model training.
- **Feature Engineering**: Craft new features to enrich the dataset and potentially improve model performance.
- **Model Selection and Training**: Compare various machine learning algorithms like Logistic Regression, Random Forest, and XGBoost to identify the best fit for predicting survival.
- **Model Evaluation and Analysis**: Assess the chosen model's accuracy, understand its strengths and weaknesses, and interpret the importance of different features in influencing survival.
- **Visualization and Insights**: Present data insights and model performance through interactive visualizations and clear explanations.

## Repository Contents:

- **notebooks**: Jupyter notebooks containing data exploration, model training, and evaluation code.
- **data**: Folder containing the Titanic passenger dataset.
- **models**: Folder containing trained machine learning models.
- **README.md**: This file (presenting the project overview).
"""

# Create documentation popup
with st.expander("📚 README.md"):
    st.markdown(readme_content)

# Layout: First Row with README and History together
st.subheader("Titanic Disaster: Historical Context")
col_history1, col_history2 = st.columns([3, 2])

with col_history1:
    st.markdown("""
    ### About the Disaster
    The RMS Titanic sank on April 15, 1912, after colliding with an iceberg during her maiden voyage 
    from Southampton to New York City. Of the estimated 2,224 passengers and crew aboard, more than 
    1,500 died, making it one of the deadliest commercial peacetime maritime disasters in modern history.
    
    ### Survival Factors
    - **Women and children first**: Following maritime tradition, women and children were given priority for lifeboats
    - **Class**: First-class passengers had better access to lifeboats
    - **Age**: Children were more likely to be saved
    - **Family size**: Passengers traveling alone had different survival patterns than those with family
    """)

with col_history2:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/640px-RMS_Titanic_3.jpg", 
             caption="RMS Titanic departing Southampton on April 10, 1912")

# Horizontal line to separate history from input form
st.markdown("---")

# Second Row with input options
st.subheader("Enter Passenger Information")
col1, col2, col3 = st.columns(3)

# Input parameters in the first column
with col1:
    st.markdown("#### Passenger Details")
    # Age input
    age = st.number_input("Age", min_value=1, max_value=80, value=30)
    
    # Fare input - Corrected to actual maximum from the Titanic dataset (approx £512)
        # Fare input with historically accurate maximum
    fare = st.number_input("Fare (£)", min_value=0.0, max_value=512.0, value=32.0, step=1.0, 
                          help="Ticket price in 1912 pounds - the most expensive actual fare was £512 for a first-class parlor suite")

# More input parameters in the second column
with col2:
    st.markdown("#### Travel Class")
    # Class selection with explanation tooltip
    pclass = st.selectbox("Passenger Class", ["1st Class", "2nd Class", "3rd Class"],
                         help="Passenger class determined cabin location and access to lifeboats, affecting survival rates")
    
    # Sex selection
    sex = st.radio("Gender", ["Male", "Female"])

# More input parameters in the third column
with col3:
    st.markdown("#### Family & Embarkation")
    # Family members
    sibsp = st.number_input("Number of Siblings/Spouses aboard", min_value=0, max_value=8, value=0)
    parch = st.number_input("Number of Parents/Children aboard", min_value=0, max_value=6, value=0)
    
    # Embarked selection
    embarked = st.selectbox("Port of Embarkation", 
                           ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])

# Centered prediction button
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn2:
    predict_button = st.button("Predict Survival", type="primary", use_container_width=True)

# Results section - placed directly below inputs without any extra space
if predict_button:
    # Prepare input data and make prediction before displaying results
    input_df = prepare_input_data()
    survival_prob = dt_model.predict_proba(input_df)[0][1]
    survival_prediction = dt_model.predict(input_df)[0]
    
    # Display results immediately below the inputs
    st.markdown("---")
    st.header("Prediction Results")
    
    # Create two columns for quick summary
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.markdown("### 👤 Passenger Summary")
        st.markdown(f"""
        Age: {age} years  
        Ticket Class: {pclass}  
        Gender: {sex}  
        Fare: £{fare}  
        Family: {sibsp} siblings/spouses, {parch} parents/children  
        Embarked at: {embarked}
        """)
    with res_col2:
        st.markdown("### 🎯 Prediction Result")
        st.progress(float(survival_prob))
        st.metric("Survival Probability", f"{survival_prob:.1%}")
        if survival_prediction == 1:
            st.success("Likely SURVIVE")
        else:
            st.error("Likely NOT SURVIVE")

    # Explanations and global importance as sections (no tabs)
    st.markdown("---")
    st.subheader("AI Explainability")
    st.markdown(
        """
        These tools explain the model's prediction in human terms:
        - SHAP shows how each feature pushed this single prediction higher or lower.
        - LIME approximates the model near this one passenger and shows which features mattered most.
        - Global Importance shows which features are most important across the whole model.
        """
    )
    
    st.subheader("Explanation (SHAP)")
    if not _shap_available:
        st.warning("SHAP is not installed. Install with: pip install shap")
    elif bg_X is None or len(bg_X) == 0:
        st.info("Background data not available to compute SHAP explanations.")
    else:
        shap_explainer = get_shap_explainer(dt_model, bg_X)
        if shap_explainer is None:
            st.info("Unable to initialize SHAP explainer for this model.")
        else:
            try:
                shap_values = shap_explainer(input_df)
                # Normalize to single-output explanation (pick class 1 when available)
                sv_single = shap_values
                try:
                    if hasattr(sv_single, "values") and getattr(sv_single.values, "ndim", 0) == 3:
                        class_index = 1 if sv_single.values.shape[2] > 1 else 0
                        base_vals = (
                            sv_single.base_values[:, class_index]
                            if np.ndim(sv_single.base_values) == 2 else sv_single.base_values
                        )
                        sv_single = shap.Explanation(
                            values=sv_single.values[:, :, class_index],
                            base_values=base_vals,
                            data=sv_single.data,
                            feature_names=friendly_feature_names
                        )
                except Exception:
                    pass
                # Ensure friendly names even when not multi-output
                try:
                    if not hasattr(sv_single, "feature_names") or sv_single.feature_names is None:
                        sv_single = shap.Explanation(
                            values=sv_single.values,
                            base_values=sv_single.base_values,
                            data=sv_single.data,
                            feature_names=friendly_feature_names
                        )
                except Exception:
                    pass
                st.markdown("#### Local feature contributions")
                fig, ax = plt.subplots(figsize=(8, 5))
                try:
                    shap.plots.waterfall(sv_single[0], show=False)
                    st.pyplot(plt.gcf(), clear_figure=True, use_container_width=True)
                except Exception:
                    plt.clf()
                    shap.plots.bar(sv_single, show=False)
                    st.pyplot(plt.gcf(), clear_figure=True, use_container_width=True)
                # Plain-English summary (top pushes)
                try:
                    values = np.array(sv_single.values[0])
                    names = friendly_feature_names
                    order = np.argsort(np.abs(values))[::-1]
                    top = [(names[i], float(values[i])) for i in order[:3]]
                    contrib_text = ", ".join([f"{n} ({'↑' if v>0 else '↓'} {abs(v):.3f})" for n, v in top])
                    st.markdown(f"In this prediction, the biggest pushes were: {contrib_text}.")
                except Exception:
                    pass
            except Exception as e:
                st.warning(f"SHAP explanation failed: {e}")

    st.markdown("---")
    st.subheader("Explanation (LIME)")
    if not _lime_available:
        st.warning("LIME is not installed. Install with: pip install lime")
    elif bg_X is None or len(bg_X) == 0:
        st.info("Background data not available to compute LIME explanations.")
    else:
        lime_explainer = get_lime_explainer(bg_X)
        if lime_explainer is None:
            st.info("Unable to initialize LIME explainer.")
        else:
            try:
                exp = lime_explainer.explain_instance(
                    data_row=input_df.iloc[0].values,
                    predict_fn=dt_model.predict_proba,
                    num_features=min(10, input_df.shape[1])
                )
                st.markdown("#### Local feature contributions")
                fig = exp.as_pyplot_figure()
                st.pyplot(fig, clear_figure=True, use_container_width=True)
            except Exception as e:
                st.warning(f"LIME explanation failed: {e}")
            # Plain-English summary from LIME weights
            try:
                pairs = exp.as_list()
                pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:3]
                lime_text = ", ".join([f"{p[0]} ({'↑' if p[1]>0 else '↓'} {abs(p[1]):.3f})" for p in pairs_sorted])
                st.markdown(f"Near this passenger, the most influential factors were: {lime_text}.")
            except Exception:
                pass

    st.markdown("---")
    st.subheader("Global Importance")
    try:
        importances = getattr(dt_model, 'feature_importances_', None)
        if importances is None:
            st.info("Global importances are not available for this model type.")
        else:
            importances = np.array(importances)
            order = np.argsort(importances)[::-1]
            top_k = 10
            top_idx = order[:top_k]
            top_features = [feature_names[i] for i in top_idx]
            top_values = importances[top_idx]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(range(len(top_features))[::-1], top_values[:][::-1], color="#4e79a7")
            ax.set_yticks(range(len(top_features))[::-1])
            ax.set_yticklabels(top_features[::-1])
            ax.set_xlabel('Importance')
            ax.set_title('Top Feature Importances')
            st.pyplot(fig, clear_figure=True, use_container_width=True)
            # Plain-English summary for global
            try:
                top3 = [(top_features[i], float(top_values[i])) for i in range(min(3, len(top_features)))]
                global_text = ", ".join([f"{n} ({v:.3f})" for n, v in top3])
                st.markdown(f"Across the whole model, the top drivers are: {global_text}.")
            except Exception:
                pass
    except Exception as e:
        st.warning(f"Failed to compute global importances: {e}")

# Add footer with information
st.markdown("---")
st.caption("Data source: Kaggle Titanic Dataset | Model: Decision Tree (max_depth=8, max_leaf_nodes=25)")

# Add developer credits with border that adapts to light/dark mode
st.markdown("""
<div style="text-align: center; margin-top: 12px; color: #6c757d; font-size: 14px; line-height: 1.4;">
  <span style="display: inline-block; padding: 10px 16px; border: 1px solid rgba(108,117,125,0.35); border-radius: 6px; background: rgba(0,0,0,0.02);">
    Designed and Developed by <strong>Aradhya Pavan</strong> · Decision Tree Classification Model
  </span>
</div>
""", unsafe_allow_html=True)