# malaria_app.py
from io import StringIO
import inspect

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import json
import sys
import os
from code_editor import code_editor


# Set page configuration
st.set_page_config(
    page_title="Malaria Prediction System",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
    }
    .positive {
        background-color: #FEE2E2;
        border: 2px solid #DC2626;
    }
    .negative {
        background-color: #DCFCE7;
        border: 2px solid #16A34A;
    }
    .symptom-card {
        padding: 15px;
        border-radius: 8px;
        background-color: #F3F4F6;
        margin: 10px 0;
    }
    .metric-card {
        padding: 15px;
        border-radius: 8px;
        background-color: #EFF6FF;
        margin: 10px 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# Load models and data
@st.cache_resource
def load_models():
    """Load trained models and related files"""
    try:
        # Load best model
        best_model = joblib.load('saved_models/malaria_best_model.pkl')

        # Load scaler
        scaler = joblib.load('saved_models/malaria_scaler.pkl')

        # Load feature names
        feature_names = joblib.load('saved_models/feature_names.pkl')

        # Load model results
        model_results = joblib.load('saved_models/model_results.pkl')

        # Load model summary
        with open('saved_models/model_summary.json', 'r') as f:
            model_summary = json.load(f)

        return {
            'model': best_model,
            'scaler': scaler,
            'feature_names': feature_names,
            'model_results': model_results,
            'model_summary': model_summary
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None


# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None


# Load models
def load_all_models():
    with st.spinner("Loading malaria prediction models..."):
        models_data = load_models()
        if models_data:
            st.session_state.models_loaded = True
            st.session_state.models_data = models_data
            return True
        return False


# Sidebar navigation
st.sidebar.title("🦟 Malaria Prediction System")
st.sidebar.markdown("---")

menu = ["🏠 Home", "🔍 Predict Malaria", "📊 Model Analysis", "📈 Data Insights","🛠️ Data Preprocessing", "ℹ️ About"]
choice = st.sidebar.selectbox("Navigation", menu)

# Home Page
# Home Page - Dashboard Style
if choice == "🏠 Home":

    # =========================
    # CUSTOM CSS
    # =========================
    st.markdown("""
    <style>
    /* HEADER */
    .main-header {
        text-align: center;
        color: #00695c;
        font-size: 3rem;
        font-weight: bold;
        margin: 20px 0 5px;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.15);
    }
    .subtitle {
        text-align: center;
        color: #00897b;
        font-size: 1.5rem;
        margin-bottom: 30px;
        font-style: italic;
    }

    /* INFO CARD */
    .info-card {
        background-color: #e0f2f1;
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        transition: transform 0.3s, box-shadow 0.3s;
    }
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }

    /* WELCOME SECTION */
    .welcome-section {
        background: #f5fdfd;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.06);
        margin: 20px 0;
    }
    .feature-list li, .how-to li {
        margin-bottom: 8px;
    }
    .how-to {
        background-color: #e8f5e9;
        padding: 18px;
        border-radius: 12px;
        border-left: 6px solid #66bb6a;
        margin-top: 20px;
    }

    /* STAT CARDS */
    .stat-card {
        border-radius: 15px;
        padding: 25px;
        color: white;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        transition: transform 0.3s, box-shadow 0.3s;
    }
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    .stat-icon {
        font-size: 2.2rem;
        margin-bottom: 10px;
    }

    /* COLORS */
    .cases { background: linear-gradient(135deg,#42a5f5,#1e88e5);}
    .deaths { background: linear-gradient(135deg,#ef5350,#e53935);}
    .risk { background: linear-gradient(135deg,#66bb6a,#43a047);}
    .affected { background: linear-gradient(135deg,#ab47bc,#8e24aa);}

    /* RESPONSIVE IMAGE */
    img {
        max-width: 100%;
        height: auto;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }

    @media (max-width: 768px){
        .main-header { font-size: 2.3rem; }
        .subtitle { font-size: 1.3rem; }
    }
    </style>
    """, unsafe_allow_html=True)

    # =========================
    # HEADER & SUBTITLE
    # =========================
    st.markdown('<h1 class="main-header">Malaria Prediction System</h1>', unsafe_allow_html=True)

    # =========================
    # HERO IMAGE
    # =========================
    col1, col2, col3 = st.columns([1,2,1])


    # =========================
    # DEVELOPER INFO + DISCLAIMER
    # =========================
    st.markdown("""
    <div class="info-card">
        <p style="color: #00695c; font-size:1.3rem; font-weight:bold;">
            Developer: <span style="color:#d81b60; font-style:italic;">Khairullah Ibrahim Khail</span>
        </p>
        <p style="color:#444; font-style:italic; line-height:1.6;">
            ⚠️ <strong>Disclaimer:</strong> This application uses machine learning to predict malaria infection based on symptoms and patient information. Results are informational only and should not replace professional medical advice.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # WELCOME + FEATURES
    # =========================
    st.markdown("""
    <div class="welcome-section">
        <h3 style="color:#00695c; text-align:center;">Welcome to the Malaria Prediction System</h3>
        <ul class="feature-list">
            <li>🔍 <strong>Predict Malaria</strong>: Enter symptoms and get instant prediction</li>
            <li>📊 <strong>Model Analysis</strong>: View model performance and metrics</li>
            <li>📈 <strong>Data Insights</strong>: Explore symptom patterns and correlations</li>
            <li>📱 <strong>Mobile Friendly</strong>: Works on all devices</li>
        </ul>
        <div class="how-to">
            <strong>How to use:</strong><br>
            1. Go to <strong>Predict Malaria</strong> page<br>
            2. Enter patient information and symptoms<br>
            3. Get instant prediction with probability<br>
            4. View detailed analysis and recommendations
        </div>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # MODEL LOADING
    # =========================
    st.subheader("🚀 Get Started")
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False

    col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
    with col_btn2:
        if not st.session_state.models_loaded:
            if st.button("📥 Load Prediction Models", type="primary", use_container_width=True):
                if load_all_models():
                    st.session_state.models_loaded = True
                    st.success("✅ Models loaded successfully!")
                    st.rerun()
        else:
            st.success("✅ Models are ready for predictions!")

    # =========================
    # MODEL INFORMATION
    # =========================
    if st.session_state.models_loaded:
        with st.expander("📋 Model Information", expanded=False):
            summary = st.session_state.models_data['model_summary']
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Model Type", summary['model_type'])
            col2.metric("Accuracy", f"{summary['test_accuracy']:.2%}")
            col3.metric("Precision", f"{summary['precision']:.2%}")
            col4.metric("Recall", f"{summary['recall']:.2%}")

    # =========================
    # GRADIENT STAT CARDS
    # =========================
    st.markdown("---")
    st.markdown('<h3 class="stats-header">📈 Global Malaria Statistics (2023)</h3>', unsafe_allow_html=True)
    stat_cols = st.columns(4)
    with stat_cols[0]:
        st.markdown('<div class="stat-card cases"><span class="stat-icon">🌍</span><br>249M<br>Cases</div>', unsafe_allow_html=True)
    with stat_cols[1]:
        st.markdown('<div class="stat-card deaths"><span class="stat-icon">⚰️</span><br>608K<br>Deaths</div>', unsafe_allow_html=True)
    with stat_cols[2]:
        st.markdown('<div class="stat-card risk"><span class="stat-icon">🧍‍♂️</span><br>3.3B<br>At Risk</div>', unsafe_allow_html=True)
    with stat_cols[3]:
        st.markdown('<div class="stat-card affected"><span class="stat-icon">🌍</span><br>Africa<br>Most Affected</div>', unsafe_allow_html=True)


# Prediction Page
elif choice == "🔍 Predict Malaria":
    st.markdown('<h1 class="main-header">Malaria Prediction</h1>', unsafe_allow_html=True)

    if not st.session_state.models_loaded:
        st.warning("⚠️ Models not loaded. Please load models from the Home page first.")
        if st.button("Load Models Now", type="primary"):
            if load_all_models():
                st.rerun()
    else:
        st.success("✅ Models loaded. Ready for prediction!")

        # Create two columns for input
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<h3 class="sub-header">Patient Information</h3>', unsafe_allow_html=True)

            # Age input
            age = st.number_input("Age", min_value=0, max_value=120, value=30,
                                  help="Patient's age in years")

            # Gender input
            gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
            gender_encoded = 1 if gender == "Male" else 0

        with col2:
            st.markdown('<h3 class="sub-header">Common Symptoms</h3>', unsafe_allow_html=True)

            # Common symptoms in 2 columns
            col2a, col2b = st.columns(2)
            with col2a:
                fever = st.checkbox("Fever", value=True)
                headache = st.checkbox("Headache", value=True)
                chills = st.checkbox("Chills", value=True)
                sweating = st.checkbox("Sweating", value=False)

            with col2b:
                fatigue = st.checkbox("Fatigue", value=True)
                nausea = st.checkbox("Nausea", value=False)
                vomiting = st.checkbox("Vomiting", value=False)
                muscle_pain = st.checkbox("Muscle Pain", value=True)

        # Additional symptoms in expander
        with st.expander("➕ Additional Symptoms"):
            col3a, col3b = st.columns(2)
            with col3a:
                diarrhea = st.checkbox("Diarrhea", value=False)
            with col3b:
                # You can add more symptoms here if your model supports them
                pass

        # Create symptoms dictionary
        symptoms = {
            'Age': age,
            'Gender': gender_encoded,
            'Fever': 1 if fever else 0,
            'Headache': 1 if headache else 0,
            'Chills': 1 if chills else 0,
            'Sweating': 1 if sweating else 0,
            'Fatigue': 1 if fatigue else 0,
            'Nausea': 1 if nausea else 0,
            'Vomiting': 1 if vomiting else 0,
            'Muscle_Pain': 1 if muscle_pain else 0,
            'Diarrhea': 1 if diarrhea else 0
        }

        # Display selected symptoms
        st.markdown("---")
        st.subheader("Selected Symptoms Summary")

        selected_symptoms = [symptom for symptom, value in symptoms.items()
                             if value == 1 and symptom not in ['Age', 'Gender']]

        if selected_symptoms:
            cols = st.columns(4)
            for idx, symptom in enumerate(selected_symptoms):
                with cols[idx % 4]:
                    st.markdown(f'<div class="symptom-card">✅ {symptom}</div>', unsafe_allow_html=True)
        else:
            st.info("No symptoms selected")

        # Prediction button
        st.markdown("---")
        if st.button("🔍 Predict Malaria", type="primary", width='stretch'):
            with st.spinner("Analyzing symptoms..."):
                try:
                    # Get model data
                    model = st.session_state.models_data['model']
                    scaler = st.session_state.models_data['scaler']
                    feature_names = st.session_state.models_data['feature_names']

                    # Create feature vector
                    features_df = pd.DataFrame(columns=feature_names)

                    # Initialize with zeros
                    for col in feature_names:
                        features_df.loc[0, col] = 0

                    # Update with symptoms
                    for symptom, value in symptoms.items():
                        if symptom in features_df.columns:
                            features_df.loc[0, symptom] = value

                    # Replace with:
                    features_scaled = scaler.transform(features_df)
                    features_scaled_df = pd.DataFrame(features_scaled,
                                                      columns=feature_names)  # Use your actual feature names
                    prediction = model.predict(features_scaled_df)[0]

                    # Get probabilities if available
                    probability = None
                    if hasattr(model, 'predict_proba'):
                        probability = model.predict_proba(features_scaled_df)[
                            0]  # ✅ CORRECT: Using features_scaled_df (DataFrame)


                    # Store prediction in session state
                    st.session_state.current_prediction = {
                        'result': prediction,
                        'probability': probability,
                        'symptoms': symptoms,
                        'selected_symptoms': selected_symptoms
                    }
                    st.session_state.prediction_made = True

                except Exception as e:
                    st.error(f"Prediction error: {e}")

        # Display prediction results
        if st.session_state.prediction_made and st.session_state.current_prediction:
            pred_data = st.session_state.current_prediction
            is_positive = pred_data['result'] == 1

            st.markdown("---")
            st.markdown('<h2 class="sub-header">Prediction Results</h2>', unsafe_allow_html=True)

            # Results box
            if is_positive:
                st.markdown(f"""
                <div class="prediction-box positive">
                    <h2 style="color: #DC2626; margin: 0;">🦟 MALARIA POSITIVE</h2>
                    <p style="font-size: 1.2rem; margin: 10px 0;">
                        High probability of malaria infection detected
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box negative">
                    <h2 style="color: #16A34A; margin: 0;">✅ MALARIA NEGATIVE</h2>
                    <p style="font-size: 1.2rem; margin: 10px 0;">
                        Low probability of malaria infection
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Probability visualization
            if pred_data['probability'] is not None:
                col_prob1, col_prob2, col_prob3 = st.columns([1, 2, 1])
                with col_prob2:
                    prob_positive = pred_data['probability'][1]
                    prob_negative = pred_data['probability'][0]

                    # Create gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob_positive * 100,
                        title={'text': "Malaria Probability"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))

                    fig.update_layout(height=300)
                    st.plotly_chart(fig, width='stretch')

                # Probability details
                col_detail1, col_detail2 = st.columns(2)
                with col_detail1:
                    st.metric("Probability of Malaria", f"{prob_positive:.2%}")
                with col_detail2:
                    st.metric("Probability of No Malaria", f"{prob_negative:.2%}")

            # Recommendations
            st.markdown("---")
            st.subheader("📋 Recommendations")

            if is_positive:
                st.warning("""
                ### ⚠️ Immediate Action Required:

                1. **Seek Medical Attention**: Visit a healthcare facility immediately
                2. **Get Confirmed Test**: Request a blood smear test for confirmation
                3. **Start Treatment**: If confirmed, begin antimalarial treatment as prescribed
                4. **Monitor Symptoms**: Watch for worsening symptoms like high fever, confusion, or seizures
                5. **Prevent Spread**: Use mosquito nets and repellents

                **Emergency signs requiring immediate hospitalization:**
                - High fever (>39°C)
                - Severe headache
                - Confusion or seizures
                - Difficulty breathing
                - Yellow skin/eyes (jaundice)
                """)
            else:
                st.success("""
                ### ✅ Preventive Measures:

                1. **Continue Monitoring**: Watch for new or worsening symptoms
                2. **Preventive Actions**:
                   - Use insect repellent
                   - Sleep under mosquito nets
                   - Wear long-sleeved clothing
                3. **If symptoms develop**:
                   - Retest after 48 hours if symptoms persist
                   - Seek medical advice if condition worsens
                4. **High-risk areas**: Consider prophylactic medication if traveling
                """)

            # Export option
            st.markdown("---")
            col_exp1, col_exp2, col_exp3 = st.columns([2, 1, 2])
            with col_exp2:
                if st.button("📥 Export Report", width='stretch'):
                    # Create report data
                    report = {
                        "patient_info": {
                            "age": age,
                            "gender": gender
                        },
                        "symptoms": symptoms,
                        "prediction": "Positive" if is_positive else "Negative",
                        "probability_positive": float(prob_positive) if pred_data['probability'] is not None else None,
                        "probability_negative": float(prob_negative) if pred_data['probability'] is not None else None,
                        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    # Convert to DataFrame for download
                    report_df = pd.DataFrame([report])
                    csv = report_df.to_csv(index=False)

                    st.download_button(
                        label="Download CSV Report",
                        data=csv,
                        file_name="malaria_prediction_report.csv",
                        mime="text/csv"
                    )

# Model Analysis Page
elif choice == "📊 Model Analysis":
    st.markdown('<h1 class="main-header">Model Analysis</h1>', unsafe_allow_html=True)

    if not st.session_state.models_loaded:
        st.warning("⚠️ Models not loaded. Please load models from the Home page first.")
    else:
        models_data = st.session_state.models_data
        summary = models_data['model_summary']
        model_results = models_data['model_results']

        # Model Performance Overview
        st.subheader("📈 Model Performance Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Best Model", summary['model_type'])
        with col2:
            st.metric("Test Accuracy", f"{summary['test_accuracy']:.2%}")
        with col3:
            st.metric("Precision", f"{summary['precision']:.2%}")
        with col4:
            st.metric("Recall", f"{summary['recall']:.2%}")

        # Confusion Matrix
        st.subheader("🎯 Confusion Matrix")

        cm = np.array(summary['confusion_matrix'])

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

        # Model Comparison
        st.subheader("📊 Model Comparison")

        # Extract model performances
        model_comparison = []
        for model_name, result in model_results.items():
            if result is not None:
                model_comparison.append({
                    'Model': model_name,
                    'Train Accuracy': result['train_accuracy'],
                    'Test Accuracy': result['test_accuracy'],
                    'Overfitting': abs(result['train_accuracy'] - result['test_accuracy'])
                })

        if model_comparison:
            comparison_df = pd.DataFrame(model_comparison)
            comparison_df = comparison_df.sort_values('Test Accuracy', ascending=False)

            # Display table
            st.dataframe(comparison_df.style.highlight_max(subset=['Test Accuracy'], color='lightgreen'),
                         width='stretch')

            # Plot comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(comparison_df))
            width = 0.35

            ax.bar(x - width / 2, comparison_df['Train Accuracy'], width, label='Train', color='skyblue')
            ax.bar(x + width / 2, comparison_df['Test Accuracy'], width, label='Test', color='lightcoral')

            ax.set_xlabel('Model')
            ax.set_ylabel('Accuracy')
            ax.set_title('Train vs Test Accuracy by Model')
            ax.set_xticks(x)
            ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

        # Feature Importance
        st.subheader("🔍 Feature Importance")

        model = models_data['model']
        feature_names = models_data['feature_names']

        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            # Plot top features
            fig, ax = plt.subplots(figsize=(10, 8))
            top_features = feature_importance.head(15)
            ax.barh(top_features['Feature'], top_features['Importance'], color='lightcoral')
            ax.set_xlabel('Importance Score')
            ax.set_title('Top 15 Most Important Features')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

            # Display table
            st.dataframe(feature_importance, width='stretch')

        elif hasattr(model, 'coef_'):
            coefficients = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': model.coef_[0],
                'Abs_Coefficient': np.abs(model.coef_[0])
            }).sort_values('Abs_Coefficient', ascending=False)

            # Plot coefficients
            fig, ax = plt.subplots(figsize=(10, 8))
            top_coeff = coefficients.head(15)
            colors = ['red' if coef < 0 else 'blue' for coef in top_coeff['Coefficient']]
            ax.barh(top_coeff['Feature'], top_coeff['Coefficient'], color=colors)
            ax.set_xlabel('Coefficient Value')
            ax.set_title('Top 15 Feature Coefficients')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

            # Display table
            st.dataframe(coefficients, width='stretch')

# Data Insights Page
elif choice == "📈 Data Insights":
    st.markdown('<h1 class="main-header">Data Insights</h1>', unsafe_allow_html=True)

    if not st.session_state.models_loaded:
        st.warning("⚠️ Models not loaded. Please load models from the Home page first.")
    else:
        # Symptom Frequency Analysis
        st.subheader("📊 Symptom Frequency in Malaria Cases")

        # You can add your actual data analysis here
        # For now, let's create some example insights

        col1, col2 = st.columns(2)

        with col1:
            # Symptom prevalence chart (example data)
            symptoms_data = {
                'Fever': 85,
                'Chills': 78,
                'Headache': 72,
                'Fatigue': 65,
                'Muscle Pain': 58,
                'Nausea': 45,
                'Sweating': 42,
                'Vomiting': 38,
                'Diarrhea': 25
            }

            symptoms_df = pd.DataFrame(list(symptoms_data.items()),
                                       columns=['Symptom', 'Percentage'])
            symptoms_df = symptoms_df.sort_values('Percentage', ascending=True)

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(symptoms_df['Symptom'], symptoms_df['Percentage'], color='steelblue')
            ax.set_xlabel('Percentage of Malaria Cases (%)')
            ax.set_title('Symptom Prevalence in Malaria')
            ax.set_xlim(0, 100)
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

        with col2:
            # Age distribution (example)
            st.markdown("### 👥 Age Distribution")

            # Create example age distribution
            age_groups = ['0-5', '6-15', '16-30', '31-50', '51+']
            malaria_cases = [15, 25, 35, 20, 5]

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(malaria_cases, labels=age_groups, autopct='%1.1f%%',
                   colors=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFD700'])
            ax.set_title('Malaria Cases by Age Group')

            st.pyplot(fig)

        # Risk Factors Analysis
        st.subheader("⚠️ Risk Factors Analysis")

        risk_factors = {
            'Travel to endemic area': 65,
            'No mosquito net use': 58,
            'Evening outdoor activity': 45,
            'No insect repellent': 52,
            'Living near water bodies': 38,
            'Previous malaria infection': 42
        }

        risk_df = pd.DataFrame(list(risk_factors.items()),
                               columns=['Risk Factor', 'Percentage'])
        risk_df = risk_df.sort_values('Percentage', ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(risk_df['Risk Factor'], risk_df['Percentage'], color='salmon')
        ax.set_xlabel('Percentage Increase in Risk (%)')
        ax.set_title('Malaria Risk Factors')
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)

        # Seasonal Pattern
        st.subheader("📅 Seasonal Patterns")

        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        cases = [45, 38, 52, 68, 85, 92, 88, 95, 78, 62, 48, 40]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(months, cases, marker='o', linewidth=2, color='darkgreen')
        ax.fill_between(months, cases, alpha=0.3, color='lightgreen')
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Cases')
        ax.set_title('Monthly Malaria Case Pattern')
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)

        # Geographic Distribution (example)
        st.subheader("🗺️ Geographic Hotspots")

        hotspots = {
            'Sub-Saharan Africa': 94,
            'South-East Asia': 3,
            'Eastern Mediterranean': 2,
            'Americas': 0.5,
            'Western Pacific': 0.5
        }

        hotspots_df = pd.DataFrame(list(hotspots.items()),
                                   columns=['Region', 'Percentage'])

        fig = px.pie(hotspots_df, values='Percentage', names='Region',
                     title='Global Malaria Distribution by Region',
                     color_discrete_sequence=px.colors.sequential.RdBu)

        st.plotly_chart(fig, width='stretch')

# ==================Data Preprocessing Page ===========================
elif choice == "🛠️ Data Preprocessing":
    st.markdown('<h1 class="main-header">🛠️ Data Preprocessing</h1>', unsafe_allow_html=True)


    col1, col2 = st.columns([2, 1])

    df = pd.read_csv(
        'https://raw.githubusercontent.com/Sahil31312/Malaria_Prediction/refs/heads/main/realistic_malaria_dataset.csv')

    with col1:
        # ==================Step 1: Import all required libraries =================

            # Use inspect.cleandoc to remove leading whitespace automatically
        st.write("### 1 Import all required libraries")
        st.code(""" 
# Import all required libraries
import numpy as np
import pandas as pd
mport matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
print("Libraries imported successfully!")
                   """)



    def library():
        # Import all required libraries
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        import warnings
        warnings.filterwarnings('ignore')
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        st.code("Libraries imported successfully!")


    def loadDataset():
        st.write("### 2 Load and Explore Dataset")
        # Load your dataset - CHANGE THE FILENAME TO YOUR ACTUAL FILE
              # Replace with your filename
        st.code(""" 
# Load your dataset - CHANGE THE FILENAME TO YOUR ACTUAL FILE
df = pd.read_csv('https://raw.githubusercontent.com/Sahil31312/Malaria_Prediction/refs/heads/main/realistic_malaria_dataset.csv')
')  # Replace with your filename

# Display first few rows
print("First 5 rows:")
display(df.head())

# Check dataset info
print("\nDataset Info:")
print(df.info())

# Check all column names
print("\nAll Column Names:")
print(df.columns.tolist())

# Check unique values in each column
print("\nUnique values in each column:")
for col in df.columns:
    unique_vals = df[col].unique()
    print(f"{col}: {len(unique_vals)} unique values - Sample: {unique_vals[:5] if len(unique_vals) > 5 else unique_vals}")
                                        """)

        # Display first few rows
        st.code("First 5 rows:")
        st.write(df.head())

        # Check dataset info
        st.code("\nDataset Info:")
        st.code(df.info())

        # Check all column names
        st.code("\nAll Column Names:")
        st.code(df.columns.tolist())

        # Check unique values in each column
        st.code("\nUnique values in each column:")
        for col in df.columns:
            unique_vals = df[col].unique()
            st.code(
                f"{col}: {len(unique_vals)} unique values - Sample: {unique_vals[:5] if len(unique_vals) > 5 else unique_vals}")




    def CheckDataTypes():
        # Check data types
        st.write("### 3 Check Data Types and Non-Numeric Columns")
        st.code(""" 
 # Check data types
print("Data Types:")
print(df.dtypes)

# Identify non-numeric columns
non_numeric_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"\nNon-numeric columns: {non_numeric_cols}")

# Check for categorical columns that should be numeric
print("\nSample values from non-numeric columns:")
for col in non_numeric_cols[:3]:  # Show first 3
    print(f"{col}: {df[col].unique()[:10]}")
                         """)
        st.code("Data Types:")

        st.code(df.dtypes)

        # Identify non-numeric columns
        non_numeric_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        st.code(f"\nNon-numeric columns: {non_numeric_cols}")

        # Check for categorical columns that should be numeric
        st.code("\nSample values from non-numeric columns:")
        for col in non_numeric_cols[:3]:  # Show first 3
            st.code(f"{col}: {df[col].unique()[:10]}")



    def HandleCategoricalVariables():
        st.write("### 4 Handle Categorical Variables")
        st.code(""" # Create a copy to work with
df_processed = df.copy()

# If there are categorical variables, let's handle them
if non_numeric_cols:
    print("Handling categorical variables...")
    
    # Use LabelEncoder for categorical columns
    label_encoders = {}
    
    for col in non_numeric_cols:
        # Don't encode if it's already the target variable or if it's supposed to be numeric
        if col != 'Positive_Malaria':
            print(f"Encoding column: {col}")
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
            print(f"  Encoded values mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Check if Positive_Malaria needs encoding
    if 'Positive_Malaria' in non_numeric_cols:
        print("Encoding target variable 'Positive_Malaria'...")
        le_target = LabelEncoder()
        df_processed['Positive_Malaria'] = le_target.fit_transform(df_processed['Positive_Malaria'].astype(str))
        print(f"  Mapping: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")
else:
    print("No categorical variables found (all are numeric)")""")

        # Create a copy to work with
        df_processed = df.copy()

        st.session_state.df_processed = df_processed


        non_numeric_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        # If there are categorical variables, let's handle them
        if non_numeric_cols:
            st.code("Handling categorical variables...")

            # Use LabelEncoder for categorical columns
            label_encoders = {}

            for col in non_numeric_cols:
                # Don't encode if it's already the target variable or if it's supposed to be numeric
                if col != 'Positive_Malaria':
                    st.code(f"Encoding column: {col}")
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    label_encoders[col] = le
                    st.code(f"  Encoded values mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

            # Check if Positive_Malaria needs encoding
            if 'Positive_Malaria' in non_numeric_cols:
                st.code("Encoding target variable 'Positive_Malaria'...")
                le_target = LabelEncoder()
                st.session_state.le_target=le_target
                df_processed['Positive_Malaria'] = le_target.fit_transform(df_processed['Positive_Malaria'].astype(str))
                st.code(f"  Mapping: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

        else:
            st.code("No categorical variables found (all are numeric)")







    def CheckforMissingValues():
        st.write("### 5 Check For Missing Values")
        st.code("""# Check for missing values
print("Missing values in each column:")
missing_values = df_processed.isnull().sum()
print(missing_values)

# Visualize missing values if any
if missing_values.sum() > 0:
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_processed.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.show()
else:
    print("No missing values found!")""")

        # Check for missing values
        if 'df_processed' not in st.session_state:
            st.error(
                "⚠️ Processed data not found! Please run the previous preprocessing steps first (e.g., Handle Categorical Variables).")
            st.stop()  # Stops execution here until data is ready
            return
        df_processed = st.session_state.df_processed

        st.code("Missing values in each column:")
        missing_values = df_processed.isnull().sum()
        st.code(missing_values)
        st.session_state.missing_values = missing_values

        # Visualize missing values if any
        if missing_values.sum() > 0:
            plt.figure(figsize=(10, 6))
            sns.heatmap(df_processed.isnull(), yticklabels=False, cbar=False, cmap='viridis')
            plt.title('Missing Values Heatmap')
            st.pyplot(plt.gcf())
        else:
            st.code("No missing values found!")


    def HandelingMissingValues():
        st.write("### 6 Handeling Missing Values")
        st.code("""# Handle missing values if any
if missing_values.sum() > 0:
print("Handling missing values...")

# Get numeric columns
numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()

# Separate target from features
if 'Positive_Malaria' in df_processed.columns:
    target_col = 'Positive_Malaria'
    feature_cols = [col for col in numeric_cols if col != target_col]
else:
    feature_cols = numeric_cols

# Create imputer for features
imputer = SimpleImputer(strategy='median')
df_processed[feature_cols] = imputer.fit_transform(df_processed[feature_cols])

# For target variable, drop rows with missing values
if target_col in df_processed.columns and df_processed[target_col].isnull().sum() > 0:
    print(f"Dropping {df_processed[target_col].isnull().sum()} rows with missing target values")
    df_processed = df_processed.dropna(subset=[target_col])

print("Missing values after handling:")
print(df_processed.isnull().sum())""")
        # Handle missing values if any
        missing_values = st.session_state.missing_values
        if missing_values.sum() > 0:
            st.code("Handling missing values...")
            df_processed = st.session_state.df_processed
            # Get numeric columns
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()

            # Separate target from features
            if 'Positive_Malaria' in df_processed.columns:
                target_col = 'Positive_Malaria'
                feature_cols = [col for col in numeric_cols if col != target_col]
            else:
                feature_cols = numeric_cols

            # Create imputer for features
            imputer = SimpleImputer(strategy='median')
            df_processed[feature_cols] = imputer.fit_transform(df_processed[feature_cols])

            # For target variable, drop rows with missing values
            if target_col in df_processed.columns and df_processed[target_col].isnull().sum() > 0:
                st.code(f"Dropping {df_processed[target_col].isnull().sum()} rows with missing target values")
                df_processed = df_processed.dropna(subset=[target_col])

            st.code("Missing values after handling:")
            st.code(df_processed.isnull().sum())

    def CheckTargetVariableDistribution():
        st.write("### 7 Check Target Variable Distribution")
        st.code("""# Check distribution of target variable
if 'Positive_Malaria' in df_processed.columns:
    print("Target variable distribution:")
    target_counts = df_processed['Positive_Malaria'].value_counts()
    print(target_counts)
    
    # Convert to readable labels if we encoded
    if 'le_target' in locals():
        # Reverse mapping for display
        reverse_mapping = {v: k for k, v in zip(le_target.classes_, le_target.transform(le_target.classes_))}
        print(f"\nMapping: {reverse_mapping}")
    
    print(f"\nPercentage distribution:")
    print(df_processed['Positive_Malaria'].value_counts(normalize=True) * 100)
    
    # Visualize
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Positive_Malaria', data=df_processed)
    plt.title('Distribution of Malaria Cases')
    plt.xlabel('Positive_Malaria (0=Negative, 1=Positive or encoded)')
    plt.show()""")

        # Check distribution of target variable
        df_processed = st.session_state.df_processed

        if 'Positive_Malaria' in df_processed.columns:
            st.code("Target variable distribution:")
            target_counts = df_processed['Positive_Malaria'].value_counts()
            st.code(target_counts)

            # Convert to readable labels if we encoded
            if 'le_target' in locals():
                # Reverse mapping for display
                le_target =  st.session_state.le_target

                reverse_mapping = {v: k for k, v in zip(le_target.classes_, le_target.transform(le_target.classes_))}
                st.code(f"\nMapping: {reverse_mapping}")

            st.code(f"\nPercentage distribution:")
            st.code(df_processed['Positive_Malaria'].value_counts(normalize=True) * 100)

            # Visualize
            plt.figure(figsize=(8, 5))
            sns.countplot(x='Positive_Malaria', data=df_processed)
            plt.title('Distribution of Malaria Cases')
            plt.xlabel('Positive_Malaria (0=Negative, 1=Positive or encoded)')
            st.pyplot(plt.gcf())




    def NowTryCorrelationAnalysis():
        st.write("### 8 Now Try Correlation Analysis")
        st.code(""" # Now calculate correlation matrix
try:
    correlation_matrix = df_processed.corr()
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, 
                annot_kws={'size': 8})
    plt.title('Correlation Matrix of Features')
    plt.tight_layout()
    plt.show()
    
    # Check correlation with target variable
    if 'Positive_Malaria' in correlation_matrix.columns:
        print("Correlation with Positive_Malaria:")
        target_corr = correlation_matrix['Positive_Malaria'].sort_values(ascending=False)
        print(target_corr)
        
        # Plot top correlations
        top_features = target_corr.abs().sort_values(ascending=False).index[1:11]  # Exclude self
        plt.figure(figsize=(10, 6))
        target_corr[top_features].plot(kind='bar')
        plt.title('Top Features Correlated with Positive_Malaria')
        plt.ylabel('Correlation Coefficient')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
except Exception as e:
    print(f"Error in correlation analysis: {e}")
    print("Showing first few rows of processed data instead:")
    display(df_processed.head())""")

        # Now calculate correlation matrix
        try:
            df_processed = st.session_state.df_processed
            correlation_matrix = df_processed.corr()

            # Plot heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5,
                        annot_kws={'size': 8})
            plt.title('Correlation Matrix of Features')
            plt.tight_layout()
            st.pyplot(plt.gcf())

            # Check correlation with target variable
            if 'Positive_Malaria' in correlation_matrix.columns:
                st.code("Correlation with Positive_Malaria:")
                target_corr = correlation_matrix['Positive_Malaria'].sort_values(ascending=False)
                st.code(target_corr)

                # Plot top correlations
                top_features = target_corr.abs().sort_values(ascending=False).index[1:11]  # Exclude self
                plt.figure(figsize=(10, 6))
                target_corr[top_features].plot(kind='bar')
                plt.title('Top Features Correlated with Positive_Malaria')
                plt.ylabel('Correlation Coefficient')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(plt.gcf())

        except Exception as e:
            st.code(f"Error in correlation analysis: {e}")
            st.code("Showing first few rows of processed data instead:")
            st.code(df_processed.head())


    def exploreFeatureDistributions():
        st.write("### 9 Explore Feature Distributions")
        st.code(""" # Plot distributions of numeric features
numeric_features = df_processed.select_dtypes(include=[np.number]).columns.tolist()
if 'Positive_Malaria' in numeric_features:
    numeric_features.remove('Positive_Malaria')

# Plot histograms for numeric features
n_features = len(numeric_features)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
fig.suptitle('Distribution of Features', fontsize=16)

for idx, feature in enumerate(numeric_features):
    row = idx // n_cols
    col = idx % n_cols
    if n_rows > 1:
        ax = axes[row, col]
    else:
        ax = axes[col] if n_cols > 1 else axes
    
    sns.histplot(data=df_processed, x=feature, ax=ax, kde=True)
    ax.set_title(f'{feature} Distribution')
    
    # Add mean and median lines
    ax.axvline(df_processed[feature].mean(), color='red', linestyle='--', alpha=0.7, label='Mean')
    ax.axvline(df_processed[feature].median(), color='green', linestyle='-.', alpha=0.7, label='Median')
    if idx == 0:  # Add legend only once
        ax.legend()

# Hide empty subplots
for idx in range(len(numeric_features), n_rows * n_cols):
    row = idx // n_cols
    col = idx % n_cols
    if n_rows > 1:
        axes[row, col].axis('off')
    else:
        axes[col].axis('off()' if n_cols > 1 else '')

plt.tight_layout()
plt.show()""")

        # Plot distributions of numeric features
        df_processed = st.session_state.df_processed
        numeric_features = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        if 'Positive_Malaria' in numeric_features:
            numeric_features.remove('Positive_Malaria')

        # Plot histograms for numeric features
        n_features = len(numeric_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        fig.suptitle('Distribution of Features', fontsize=16)

        for idx, feature in enumerate(numeric_features):
            row = idx // n_cols
            col = idx % n_cols
            if n_rows > 1:
                ax = axes[row, col]
            else:
                ax = axes[col] if n_cols > 1 else axes

            sns.histplot(data=df_processed, x=feature, ax=ax, kde=True)
            ax.set_title(f'{feature} Distribution')

            # Add mean and median lines
            ax.axvline(df_processed[feature].mean(), color='red', linestyle='--', alpha=0.7, label='Mean')
            ax.axvline(df_processed[feature].median(), color='green', linestyle='-.', alpha=0.7, label='Median')
            if idx == 0:  # Add legend only once
                ax.legend()

        # Hide empty subplots
        for idx in range(len(numeric_features), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            if n_rows > 1:
                axes[row, col].axis('off')
            else:
                axes[col].axis('off()' if n_cols > 1 else '')

        plt.tight_layout()
        st.pyplot(fig)



    def SplitFeaturesAndTarget():
        st.write("Split Features and Target")
        st.code(""" # Separate features and target
if 'Positive_Malaria' in df_processed.columns:
    X = df_processed.drop('Positive_Malaria', axis=1)  # Features
    y = df_processed['Positive_Malaria']  # Target
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFeature columns: {X.columns.tolist()}")
    
    # Display sample of features
    print("\nSample of features:")
    display(X.head())
    
    # Display target distribution
    print("\nTarget value counts:")
    print(y.value_counts())
else:
    print("Error: 'Positive_Malaria' column not found in dataset")
    print("Available columns:", df_processed.columns.tolist())""")

        # Separate features and target
        df_processed = st.session_state.df_processed
        if 'Positive_Malaria' in df_processed.columns:
            X = df_processed.drop('Positive_Malaria', axis=1)  # Features
            y = df_processed['Positive_Malaria']  # Target

            st.session_state.X=X
            st.session_state.y = y
            st.code(f"Features shape: {X.shape}")
            st.code(f"Target shape: {y.shape}")
            st.code(f"\nFeature columns: {X.columns.tolist()}")

            # Display sample of features
            st.code("\nSample of features:")
            st.dataframe(X.head())

            # Display target distribution
            st.code("\nTarget value counts:")
            st.code(y.value_counts())
        else:
            st.code("Error: 'Positive_Malaria' column not found in dataset")
            st.code("Available columns:", df_processed.columns.tolist())

    def SplitDataTrainingTestingSets():
        st.write("### 10 Split Data Training and Testing Sets")
        st.code("""# Check if X and y are defined
if 'X' in locals() and 'y' in locals():
    # Split the data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    
    # Check class distribution
    print("\nTraining target distribution:")
    train_counts = y_train.value_counts()
    print(f"Counts:\n{train_counts}")
    print(f"Percentages:\n{y_train.value_counts(normalize=True) * 100}")
    
    print("\nTesting target distribution:")
    test_counts = y_test.value_counts()
    print(f"Counts:\n{test_counts}")
    print(f"Percentages:\n{y_test.value_counts(normalize=True) * 100}")
    
    # Visualize the split
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training distribution
    axes[0].bar(train_counts.index.astype(str), train_counts.values)
    axes[0].set_title('Training Set Distribution')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(train_counts.values):
        axes[0].text(i, v + max(train_counts.values)*0.01, str(v), ha='center')
    
    # Testing distribution
    axes[1].bar(test_counts.index.astype(str), test_counts.values)
    axes[1].set_title('Testing Set Distribution')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Count')
    for i, v in enumerate(test_counts.values):
        axes[1].text(i, v + max(test_counts.values)*0.01, str(v), ha='center')
    
    plt.tight_layout()
    plt.show()
    
else:
    print("Error: X and y are not defined. Please run previous cells first.")""")

        # Check if X and y are defined
        X = st.session_state.X
        y = st.session_state.y
        if 'X' in locals() and 'y' in locals():
            # Split the data (80% train, 20% test)


            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            st.session_state.X_train= X_train
            st.session_state.y_train= y_train
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test

            st.code(f"Training set size: {X_train.shape}")
            st.code(f"Testing set size: {X_test.shape}")

            # Check class distribution
            st.code("\nTraining target distribution:")
            train_counts = y_train.value_counts()
            st.code(f"Counts:\n{train_counts}")
            st.code(f"Percentages:\n{y_train.value_counts(normalize=True) * 100}")

            st.code("\nTesting target distribution:")
            test_counts = y_test.value_counts()
            st.code(f"Counts:\n{test_counts}")
            st.code(f"Percentages:\n{y_test.value_counts(normalize=True) * 100}")

            # Visualize the split
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Training distribution
            axes[0].bar(train_counts.index.astype(str), train_counts.values)
            axes[0].set_title('Training Set Distribution')
            axes[0].set_xlabel('Class')
            axes[0].set_ylabel('Count')
            for i, v in enumerate(train_counts.values):
                axes[0].text(i, v + max(train_counts.values) * 0.01, str(v), ha='center')

            # Testing distribution
            axes[1].bar(test_counts.index.astype(str), test_counts.values)
            axes[1].set_title('Testing Set Distribution')
            axes[1].set_xlabel('Class')
            axes[1].set_ylabel('Count')
            for i, v in enumerate(test_counts.values):
                axes[1].text(i, v + max(test_counts.values) * 0.01, str(v), ha='center')

            plt.tight_layout()
            st.pyplot(fig)

        else:
            st.code("Error: X and y are not defined. Please run previous cells first.")


    def FeatureScaling():
        st.write("### 12 Feature Scaling")

        st.code("""# Initialize scaler
scaler = StandardScaler()

# Fit on training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for better readability
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

print("Scaled training data statistics:")
print("Mean of each feature (should be ~0):")
print(X_train_scaled_df.mean().round(3))

print("\nStandard deviation of each feature (should be ~1):")
print(X_train_scaled_df.std().round(3))

print("\nScaled training data sample (first 5 rows):")
display(X_train_scaled_df.head())

# Visualize scaling effect on first few features
if len(X.columns) >= 2:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Before scaling
    for i, feature in enumerate(X.columns[:2]):
        axes[0, i].hist(X_train[feature], bins=30, alpha=0.7, color='blue', label='Original')
        axes[0, i].set_title(f'{feature} - Before Scaling')
        axes[0, i].set_xlabel('Value')
        axes[0, i].set_ylabel('Frequency')
        axes[0, i].legend()
    
    # After scaling
    for i, feature in enumerate(X.columns[:2]):
        axes[1, i].hist(X_train_scaled_df[feature], bins=30, alpha=0.7, color='green', label='Scaled')
        axes[1, i].set_title(f'{feature} - After Scaling')
        axes[1, i].set_xlabel('Value')
        axes[1, i].set_ylabel('Frequency')
        axes[1, i].legend()
    
    plt.suptitle('Feature Scaling Effect', fontsize=16)
    plt.tight_layout()
    plt.show()""")

        # Initialize scaler
        X = st.session_state.X
        y = st.session_state.y
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test

        scaler = StandardScaler()
        st.session_state.scaler = scaler

        # Fit on training data and transform both training and testing data
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Convert back to DataFrame for better readability
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

        st.code("Scaled training data statistics:")
        st.code("Mean of each feature (should be ~0):")
        st.code(X_train_scaled_df.mean().round(3))

        st.code("\nStandard deviation of each feature (should be ~1):")
        st.code(X_train_scaled_df.std().round(3))

        st.code("\nScaled training data sample (first 5 rows):")
        st.code(X_train_scaled_df.head())

        # ========= sessiont data ============
        st.session_state.X_train_scaled_df = X_train_scaled_df
        st.session_state.X_test_scaled_df = X_test_scaled_df

        # Visualize scaling effect on first few features
        if len(X.columns) >= 2:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Before scaling
            for i, feature in enumerate(X.columns[:2]):
                axes[0, i].hist(X_train[feature], bins=30, alpha=0.7, color='blue', label='Original')
                axes[0, i].set_title(f'{feature} - Before Scaling')
                axes[0, i].set_xlabel('Value')
                axes[0, i].set_ylabel('Frequency')
                axes[0, i].legend()

            # After scaling
            for i, feature in enumerate(X.columns[:2]):
                axes[1, i].hist(X_train_scaled_df[feature], bins=30, alpha=0.7, color='green', label='Scaled')
                axes[1, i].set_title(f'{feature} - After Scaling')
                axes[1, i].set_xlabel('Value')
                axes[1, i].set_ylabel('Frequency')
                axes[1, i].legend()

            plt.suptitle('Feature Scaling Effect', fontsize=16)
            plt.tight_layout()
            st.pyplot(fig)











    def TrainMultipleModels():
        st.write("### 13 Train Multiple Models")
        st.code(""" # Dictionary to store models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(random_state=42, probability=True),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Dictionary to store results
results = {}
predictions = {}

# Train each model
print("Training models...")
print("=" * 60)

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    try:
        # Train model
        model.fit(X_train_scaled_df, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled_df)
        y_pred_train = model.predict(X_train_scaled_df)
        
        # Calculate accuracies
        test_accuracy = accuracy_score(y_test, y_pred)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        
        # Store results
        results[name] = {
            'model': model,
            'test_accuracy': test_accuracy,
            'train_accuracy': train_accuracy,
            'y_pred': y_pred,
            'predictions_train': y_pred_train
        }
        
        predictions[name] = y_pred
        
        print(f"  Training Accuracy: {train_accuracy:.4f}")
        print(f"  Testing Accuracy:  {test_accuracy:.4f}")
        print(f"  Overfitting (Train-Test diff): {abs(train_accuracy - test_accuracy):.4f}")
        
    except Exception as e:
        print(f"  Error training {name}: {e}")
        results[name] = None

print("\n" + "=" * 60)
print("All models trained successfully!")""")

        # Dictionary to store models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }

        st.session_state.models = models

        # Dictionary to store results
        results = {}
        predictions = {}

        # Train each model
        st.code("Training models...")

        X_train_scaled_df = st.session_state.X_train_scaled_df
        X_test_scaled_df = st.session_state.X_test_scaled_df
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test

        for name, model in models.items():
            st.write(f"\nTraining {name}...")

            try:
                # Train model
                model.fit(X_train_scaled_df, y_train)

                # Make predictions
                y_pred = model.predict(X_test_scaled_df)
                y_pred_train = model.predict(X_train_scaled_df)

                # Calculate accuracies
                test_accuracy = accuracy_score(y_test, y_pred)
                train_accuracy = accuracy_score(y_train, y_pred_train)

                # Store results
                results[name] = {
                    'model': model,
                    'test_accuracy': test_accuracy,
                    'train_accuracy': train_accuracy,
                    'y_pred': y_pred,
                    'predictions_train': y_pred_train
                }

                #  ============  session data ============
                st.session_state.results = results

                predictions[name] = y_pred

                st.code(f"  Training Accuracy: {train_accuracy:.4f}")
                st.code(f"  Testing Accuracy:  {test_accuracy:.4f}")
                st.code(f"  Overfitting (Train-Test diff): {abs(train_accuracy - test_accuracy):.4f}")

            except Exception as e:
                st.code(f"  Error training {name}: {e}")
                results[name] = None


        st.write("All models trained successfully!")


    def CompareModelPerformance():
        st.write("### 14 Compare Model Performance")

        st.code(""" # Create comparison dataframe
performance_data = []
for name, result in results.items():
    if result is not None:
        performance_data.append({
            'Model': name,
            'Train_Accuracy': result['train_accuracy'],
            'Test_Accuracy': result['test_accuracy'],
            'Overfitting': abs(result['train_accuracy'] - result['test_accuracy'])
        })

if performance_data:
    results_df = pd.DataFrame(performance_data)
    results_df = results_df.sort_values('Test_Accuracy', ascending=False).reset_index(drop=True)
    
    print("Model Performance Comparison:")
    display(results_df)
    
    # Visualize model performance
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Test Accuracy comparison
    bars1 = axes[0].barh(results_df['Model'], results_df['Test_Accuracy'], color='lightblue')
    axes[0].set_xlabel('Test Accuracy')
    axes[0].set_title('Model Test Accuracy Comparison')
    axes[0].set_xlim([0, 1])
    axes[0].invert_yaxis()
    
    # Add accuracy values on bars
    for bar in bars1:
        width = bar.get_width()
        axes[0].text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', ha='left', va='center')
    
    # Train vs Test Accuracy
    x = np.arange(len(results_df))
    width = 0.35
    
    axes[1].bar(x - width/2, results_df['Train_Accuracy'], width, label='Train Accuracy', color='lightgreen')
    axes[1].bar(x + width/2, results_df['Test_Accuracy'], width, label='Test Accuracy', color='lightblue')
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Train vs Test Accuracy')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(results_df['Model'], rotation=45, ha='right')
    axes[1].legend()
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.show()
    
    # Identify best model
    best_model_name = results_df.iloc[0]['Model']
    best_model_info = results[best_model_name]
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Best Model Test Accuracy: {results_df.iloc[0]['Test_Accuracy']:.4f}")
    
else:
    print("No models were successfully trained.") """)

        # Create comparison dataframe
        results = st.session_state.results
        performance_data = []
        for name, result in results.items():
            if result is not None:
                performance_data.append({
                    'Model': name,
                    'Train_Accuracy': result['train_accuracy'],
                    'Test_Accuracy': result['test_accuracy'],
                    'Overfitting': abs(result['train_accuracy'] - result['test_accuracy'])
                })

        if performance_data:
            results_df = pd.DataFrame(performance_data)
            results_df = results_df.sort_values('Test_Accuracy', ascending=False).reset_index(drop=True)

            st.code("Model Performance Comparison:")
            st.dataframe(results_df)

            # Visualize model performance
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Test Accuracy comparison
            bars1 = axes[0].barh(results_df['Model'], results_df['Test_Accuracy'], color='lightblue')
            axes[0].set_xlabel('Test Accuracy')
            axes[0].set_title('Model Test Accuracy Comparison')
            axes[0].set_xlim([0, 1])
            axes[0].invert_yaxis()

            # Add accuracy values on bars
            for bar in bars1:
                width = bar.get_width()
                axes[0].text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                             f'{width:.4f}', ha='left', va='center')

            # Train vs Test Accuracy
            x = np.arange(len(results_df))
            width = 0.35

            axes[1].bar(x - width / 2, results_df['Train_Accuracy'], width, label='Train Accuracy', color='lightgreen')
            axes[1].bar(x + width / 2, results_df['Test_Accuracy'], width, label='Test Accuracy', color='lightblue')
            axes[1].set_xlabel('Model')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Train vs Test Accuracy')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(results_df['Model'], rotation=45, ha='right')
            axes[1].legend()
            axes[1].set_ylim([0, 1])

            plt.tight_layout()
            st.pyplot(fig)

            # Identify best model
            best_model_name = results_df.iloc[0]['Model']
            best_model_info = results[best_model_name]

            # ============== session data  =============
            st.session_state.best_model_name = best_model_name


            st.code(f"\nBest Model: {best_model_name}")
            st.code(f"Best Model Test Accuracy: {results_df.iloc[0]['Test_Accuracy']:.4f}")

        else:
            st.code("No models were successfully trained.")

    def DetailedEvaluationofBestModel():
        st.write("Detailed Evaluation of Best Model")

        st.code("""# Get best model
if 'best_model_name' in locals() and best_model_name in results:
    best_model_info = results[best_model_name]
    best_model = best_model_info['model']
    
    print(f"Detailed Evaluation of Best Model: {best_model_name}")
    print("=" * 60)
    
    # Make predictions
    y_pred = best_model_info['y_pred']
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Plot confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Numerical confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Confusion Matrix - Numerical')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # Normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens', ax=axes[1])
    axes[1].set_title('Confusion Matrix - Normalized')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()
    
    # Classification Report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    # Calculate additional metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nKey Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Check for probability predictions
    if hasattr(best_model, 'predict_proba'):
        y_pred_proba = best_model.predict_proba(X_test_scaled_df)[:, 1]
        print(f"\nModel supports probability predictions")
    else:
        print(f"\nModel does not support probability predictions")
        
else:
    print("Best model not found. Please run previous cells first.") """)

        # Get best model
        results = st.session_state.results
        best_model_name = st.session_state.best_model_name

        if 'best_model_name' in locals() and best_model_name in results:
            best_model_info = results[best_model_name]
            best_model = best_model_info['model']

            st.session_state.best_model = best_model

            y_test = st.session_state.y_test
            X_test_scaled_df = st.session_state.X_test_scaled_df

            st.code(f"Detailed Evaluation of Best Model: {best_model_name}")


            # Make predictions
            y_pred = best_model_info['y_pred']

            # Confusion Matrix
            st.code("\nConfusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            st.code(cm)

            # Plot confusion matrix
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Numerical confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
            axes[0].set_title('Confusion Matrix - Numerical')
            axes[0].set_xlabel('Predicted')
            axes[0].set_ylabel('Actual')

            # Normalized confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens', ax=axes[1])
            axes[1].set_title('Confusion Matrix - Normalized')
            axes[1].set_xlabel('Predicted')
            axes[1].set_ylabel('Actual')

            plt.tight_layout()
            st.pyplot(fig)

            # Classification Report
            st.code("\nDetailed Classification Report:")
            st.code(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

            # Calculate additional metrics
            from sklearn.metrics import precision_score, recall_score, f1_score

            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            st.code(f"\nKey Metrics:")
            st.code(f"Precision: {precision:.4f}")
            st.code(f"Recall: {recall:.4f}")
            st.code(f"F1-Score: {f1:.4f}")

            # Check for probability predictions
            if hasattr(best_model, 'predict_proba'):
                y_pred_proba = best_model.predict_proba(X_test_scaled_df)[:, 1]
                st.code(f"\nModel supports probability predictions")
            else:
                st.code(f"\nModel does not support probability predictions")

        else:
            st.code("Best model not found. Please run previous cells first.")


    def FeatureImportanceAnalysis():
        st.write("### 16 Feature Importance Analysis")
        st.code(""" # Check if best model supports feature importance
if 'best_model' in locals():
    print(f"Feature Importance Analysis for {best_model_name}")
    print("=" * 60)
    
    if hasattr(best_model, 'feature_importances_'):
        # Tree-based models (Random Forest, Decision Tree, Gradient Boosting)
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("Feature Importance Scores:")
        display(feature_importance)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        bars = plt.barh(feature_importance['Feature'][:15],  # Top 15 features
                       feature_importance['Importance'][:15],
                       color='lightcoral')
        plt.xlabel('Importance Score')
        plt.title(f'Top 15 Feature Importance - {best_model_name}')
        plt.gca().invert_yaxis()
        
        # Add values on bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.4f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
    elif hasattr(best_model, 'coef_'):  
        # Linear models (Logistic Regression)
        coefficients = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': best_model.coef_[0],
            'Abs_Coefficient': np.abs(best_model.coef_[0])
        }).sort_values('Abs_Coefficient', ascending=False)
        
        print("Feature Coefficients:")
        display(coefficients)
        
        # Plot coefficients
        plt.figure(figsize=(10, 6))
        top_features = coefficients.head(15)
        colors = ['red' if coef < 0 else 'blue' for coef in top_features['Coefficient']]
        bars = plt.barh(top_features['Feature'], top_features['Coefficient'], color=colors)
        plt.xlabel('Coefficient Value')
        plt.title(f'Top 15 Feature Coefficients - {best_model_name}')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.gca().invert_yaxis()
        
        # Add values on bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width + (0.01 if width >= 0 else -0.01), 
                    bar.get_y() + bar.get_height()/2,
                    f'{width:.4f}', 
                    ha='left' if width >= 0 else 'right', 
                    va='center', 
                    fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
    else:
        print(f"{best_model_name} does not provide feature importance or coefficients.")
        
    # Correlation with target (alternative feature importance)
    print("\nFeature Correlation with Target (Alternative View):")
    if 'X' in locals() and 'y' in locals():
        # Calculate correlation for all features
        df_corr = pd.concat([X, y.rename('Target')], axis=1)
        corr_with_target = df_corr.corr()['Target'].drop('Target').sort_values(ascending=False)
        
        corr_df = pd.DataFrame({
            'Feature': corr_with_target.index,
            'Correlation': corr_with_target.values,
            'Abs_Correlation': np.abs(corr_with_target.values)
        }).sort_values('Abs_Correlation', ascending=False)
        
        display(corr_df.head(10))
        
        # Plot top correlations
        plt.figure(figsize=(10, 6))
        top_corr = corr_df.head(10)
        colors = ['red' if corr < 0 else 'blue' for corr in top_corr['Correlation']]
        bars = plt.barh(top_corr['Feature'], top_corr['Correlation'], color=colors)
        plt.xlabel('Correlation with Target')
        plt.title('Top 10 Features Correlated with Malaria Diagnosis')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.gca().invert_yaxis()
        
        # Add values on bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width + (0.01 if width >= 0 else -0.01), 
                    bar.get_y() + bar.get_height()/2,
                    f'{width:.4f}', 
                    ha='left' if width >= 0 else 'right', 
                    va='center', 
                    fontsize=9)
        
        plt.tight_layout()
        plt.show()""")

        # Check if best model supports feature importance
        best_model_name = st.session_state.best_model_name
        X = st.session_state.X

        best_model = st.session_state.best_model

        if 'best_model' in locals():
            print(f"Feature Importance Analysis for {best_model_name}")


            if hasattr(best_model, 'feature_importances_'):
                # Tree-based models (Random Forest, Decision Tree, Gradient Boosting)
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': best_model.feature_importances_
                }).sort_values('Importance', ascending=False)

                st.code("Feature Importance Scores:")
                st.code(feature_importance)

                # Plot feature importance
                plt.figure(figsize=(10, 6))
                bars = plt.barh(feature_importance['Feature'][:15],  # Top 15 features
                                feature_importance['Importance'][:15],
                                color='lightcoral')
                plt.xlabel('Importance Score')
                plt.title(f'Top 15 Feature Importance - {best_model_name}')
                plt.gca().invert_yaxis()

                # Add values on bars
                for bar in bars:
                    width = bar.get_width()
                    plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                             f'{width:.4f}', ha='left', va='center', fontsize=9)

                plt.tight_layout()
                st.pyplot(fig)

            elif hasattr(best_model, 'coef_'):
                # Linear models (Logistic Regression)
                coefficients = pd.DataFrame({
                    'Feature': X.columns,
                    'Coefficient': best_model.coef_[0],
                    'Abs_Coefficient': np.abs(best_model.coef_[0])
                }).sort_values('Abs_Coefficient', ascending=False)

                st.code("Feature Coefficients:")
                st.code(coefficients)

                # Plot coefficients
                plt.figure(figsize=(10, 6))
                top_features = coefficients.head(15)
                colors = ['red' if coef < 0 else 'blue' for coef in top_features['Coefficient']]
                bars = plt.barh(top_features['Feature'], top_features['Coefficient'], color=colors)
                plt.xlabel('Coefficient Value')
                plt.title(f'Top 15 Feature Coefficients - {best_model_name}')
                plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                plt.gca().invert_yaxis()

                # Add values on bars
                for bar in bars:
                    width = bar.get_width()
                    plt.text(width + (0.01 if width >= 0 else -0.01),
                             bar.get_y() + bar.get_height() / 2,
                             f'{width:.4f}',
                             ha='left' if width >= 0 else 'right',
                             va='center',
                             fontsize=9)

                plt.tight_layout()
                st.pyplot(plt.gcf())

            else:
                st.code(f"{best_model_name} does not provide feature importance or coefficients.")

            # Correlation with target (alternative feature importance)
            st.code("\nFeature Correlation with Target (Alternative View):")
            y = st.session_state.y
            if 'X' in locals() and 'y' in locals():
                # Calculate correlation for all features
                df_corr = pd.concat([X, y.rename('Target')], axis=1)
                corr_with_target = df_corr.corr()['Target'].drop('Target').sort_values(ascending=False)

                corr_df = pd.DataFrame({
                    'Feature': corr_with_target.index,
                    'Correlation': corr_with_target.values,
                    'Abs_Correlation': np.abs(corr_with_target.values)
                }).sort_values('Abs_Correlation', ascending=False)

                st.code(corr_df.head(10))

                # Plot top correlations
                plt.figure(figsize=(10, 6))
                top_corr = corr_df.head(10)
                colors = ['red' if corr < 0 else 'blue' for corr in top_corr['Correlation']]
                bars = plt.barh(top_corr['Feature'], top_corr['Correlation'], color=colors)
                plt.xlabel('Correlation with Target')
                plt.title('Top 10 Features Correlated with Malaria Diagnosis')
                plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                plt.gca().invert_yaxis()

                # Add values on bars
                for bar in bars:
                    width = bar.get_width()
                    plt.text(width + (0.01 if width >= 0 else -0.01),
                             bar.get_y() + bar.get_height() / 2,
                             f'{width:.4f}',
                             ha='left' if width >= 0 else 'right',
                             va='center',
                             fontsize=9)

                plt.tight_layout()
                st.pyplot(plt.gcf())


    def SaveTheBestModels():
        st.write("### 17 Save the Best model ")

        st.code(""" # First, let's ensure we have the best model identified
print("Checking available models and selecting the best one...")

if 'results' in locals() and len(results) > 0:
    # Find the model with highest test accuracy
    best_model_name = None
    best_accuracy = -1
    
    for model_name, model_info in results.items():
        if model_info is not None:
            accuracy = model_info['test_accuracy']
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = model_name
    
    if best_model_name:
        print(f"Best model identified: {best_model_name} with accuracy: {best_accuracy:.4f}")
        best_model = results[best_model_name]['model']
        
        # Save the best model
        import joblib
        import os
        
        # Create a directory for saving models if it doesn't exist
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
        
        # Save the model
        model_filename = 'saved_models/malaria_best_model.pkl'
        joblib.dump(best_model, model_filename)
        print(f"Best model saved as: {model_filename}")
        
        # Save the scaler
        scaler_filename = 'saved_models/malaria_scaler.pkl'
        joblib.dump(scaler, scaler_filename)
        print(f"Scaler saved as: {scaler_filename}")
        
        # Save all models
        all_models_filename = 'saved_models/all_malaria_models.pkl'
        joblib.dump(models, all_models_filename)
        print(f"All models saved as: {all_models_filename}")
        
        # Save feature names
        feature_names = X.columns.tolist()
        features_filename = 'saved_models/feature_names.pkl'
        joblib.dump(feature_names, features_filename)
        print(f"Feature names saved as: {features_filename}")
        
        # Save results
        results_filename = 'saved_models/model_results.pkl'
        joblib.dump(results, results_filename)
        print(f"Model results saved as: {results_filename}")
        
        print("\nAll files saved successfully in 'saved_models' directory!")
        
    else:
        print("No valid models found in results.")
else:
    print("Results dictionary not found or empty. Please run model training cells first.")""")

        # First, let's ensure we have the best model identified
        st.code("Checking available models and selecting the best one...")
        results = st.session_state.results
        if 'results' in locals() and len(results) > 0:
            # Find the model with highest test accuracy
            best_model_name = None
            best_accuracy = -1

            for model_name, model_info in results.items():
                if model_info is not None:
                    accuracy = model_info['test_accuracy']
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model_name = model_name

            if best_model_name:
                st.code(f"Best model identified: {best_model_name} with accuracy: {best_accuracy:.4f}")
                best_model = results[best_model_name]['model']

                # Save the best model
                import joblib
                import os


                # Create a directory for saving models if it doesn't exist
                if not os.path.exists('saved_modelsApp'):
                    os.makedirs('saved_modelsApp')

                # Save the model
                model_filename = 'saved_modelsApp/malaria_best_model.pkl'
                joblib.dump(best_model, model_filename)
                st.code(f"Best model saved as: {model_filename}")

                # Save the scaler
                scaler = st.session_state.scaler
                scaler_filename = 'saved_modelsApp/malaria_scaler.pkl'
                joblib.dump(scaler, scaler_filename)
                st.code(f"Scaler saved as: {scaler_filename}")

                # Save all models
                models = st.session_state.models
                all_models_filename = 'saved_modelsApp/all_malaria_models.pkl'
                joblib.dump(models, all_models_filename)
                st.code(f"All models saved as: {all_models_filename}")

                X  = st.session_state.X
                # Save feature names
                feature_names = X.columns.tolist()
                features_filename = 'saved_modelsApp/feature_names.pkl'
                joblib.dump(feature_names, features_filename)
                st.code(f"Feature names saved as: {features_filename}")

                # Save results
                results_filename = 'saved_modelsApp/model_results.pkl'
                joblib.dump(results, results_filename)
                st.code(f"Model results saved as: {results_filename}")

                st.code("\nAll files saved successfully in 'saved_modelsApp' directory!")

            else:
                st.code("No valid models found in results.")
        else:
            st.code("Results dictionary not found or empty. Please run model training cells first.")



    def predictionfunction():
        st.write("### 18 Prediction Function")
        st.code('''# Now test the prediction function with the fixed best_model
        print("Testing Prediction Function with Fixed Model:")
        print("=" * 60)

        if 'best_model' in locals():
            # Update the prediction function to use our fixed variables
            def predict_malaria_fixed(symptoms_dict):
                """
                Predict malaria based on symptoms

                Parameters:
                - symptoms_dict: dictionary with symptom values

                Example:
                {
                    'Age': 25,
                    'Gender': 1,  # 1 for Male, 0 for Female (after encoding)
                    'Fever': 1,
                    'Headache': 1,
                    'Chills': 1,
                    'Sweating': 0,
                    'Fatigue': 1,
                    'Nausea': 0,
                    'Vomiting': 0,
                    'Muscle_Pain': 1,
                    'Diarrhea': 0
                }
                """
                # Get global variables
                global best_model, scaler, X

                # Create a dataframe with all features
                features_df = pd.DataFrame(columns=X.columns)

                # Initialize with zeros for all features
                for col in X.columns:
                    features_df.loc[0, col] = 0

                # Update with provided symptoms
                missing_features = []
                for symptom, value in symptoms_dict.items():
                    if symptom in features_df.columns:
                        features_df.loc[0, symptom] = value
                    else:
                        missing_features.append(symptom)

                if missing_features:
                    print(f"Warning: These features were not in the model: {missing_features}")
                    print(f"Available features: {list(X.columns)}")

                # Check if all features are provided
                if len(symptoms_dict) < len(X.columns):
                    print(f"\\nNote: Only {len(symptoms_dict)} features provided out of {len(X.columns)} expected.")
                    print("Missing features will be set to 0 (default).")

                # Scale the features
                features_scaled = scaler.transform(features_df)

                # Make prediction
                prediction = best_model.predict(features_scaled)[0]

                # Get probability if available
                probability = None
                if hasattr(best_model, 'predict_proba'):
                    probability = best_model.predict_proba(features_scaled)[0]

                # Prepare result
                result = {
                    'prediction': int(prediction),
                    'prediction_label': 'Malaria Positive' if prediction == 1 else 'Malaria Negative',
                    'probability': probability,
                    'probability_positive': probability[1] if probability is not None else None,
                    'probability_negative': probability[0] if probability is not None else None,
                    'features_used': list(symptoms_dict.keys()),
                    'missing_features': [col for col in X.columns if col not in symptoms_dict],
                    'model_used': best_model_name
                }

                return result

            # Test with a complete example
            print("\\nTest 1: Complete example with all features")
            complete_example = {
                'Age': 30,
                'Gender': 1,  # Assuming 1=Male, 0=Female (check your encoding)
                'Fever': 1,
                'Headache': 1,
                'Chills': 1,
                'Sweating': 0,
                'Fatigue': 1,
                'Nausea': 0,
                'Vomiting': 0,
                'Muscle_Pain': 1,
                'Diarrhea': 0
            }

            result = predict_malaria_fixed(complete_example)

            print(f"Model used: {result['model_used']}")
            print(f"Prediction: {result['prediction_label']}")
            if result['probability_positive'] is not None:
                print(f"Probability of Malaria: {result['probability_positive']:.2%}")
                print(f"Probability of No Malaria: {result['probability_negative']:.2%}")

            # Test with partial features
            print("\\n" + "=" * 60)
            print("Test 2: Partial example (only key symptoms)")
            partial_example = {
                'Age': 25,
                'Gender': 0,  # Female
                'Fever': 1,
                'Headache': 1,
                'Chills': 1,
                'Fatigue': 1
            }

            result2 = predict_malaria_fixed(partial_example)
            print(f"\\nPrediction: {result2['prediction_label']}")
            if result2['probability_positive'] is not None:
                print(f"Probability of Malaria: {result2['probability_positive']:.2%}")
            print(f"Missing features (set to 0): {result2['missing_features']}")

            # Create a quick demo function
            print("\\n" + "=" * 60)
            print("Quick Prediction Demo:")

            def quick_predict(age, gender, fever, headache, chills):
                """Quick prediction with common symptoms"""
                symptoms = {
                    'Age': age,
                    'Gender': gender,
                    'Fever': fever,
                    'Headache': headache,
                    'Chills': chills,
                    'Sweating': 0,  # Default values for others
                    'Fatigue': 0,
                    'Nausea': 0,
                    'Vomiting': 0,
                    'Muscle_Pain': 0,
                    'Diarrhea': 0
                }

                # Update with provided values
                if fever == 1:
                    symptoms['Fever'] = 1
                if headache == 1:
                    symptoms['Headache'] = 1
                if chills == 1:
                    symptoms['Chills'] = 1

                result = predict_malaria_fixed(symptoms)
                return result

            # Example usage
            print("\\nExample 1: 25-year-old Male with Fever, Headache, and Chills")
            demo1 = quick_predict(age=25, gender=1, fever=1, headache=1, chills=1)
            print(f"Result: {demo1['prediction_label']}, Probability: {demo1['probability_positive']:.2%}")

            print("\\nExample 2: 40-year-old Female with only Headache")
            demo2 = quick_predict(age=40, gender=0, fever=0, headache=1, chills=0)
            print(f"Result: {demo2['prediction_label']}, Probability: {demo2['probability_positive']:.2%}")

        else:
            print("best_model not found. Please run previous cells first.")''', language='python')

        # Now test the prediction function with the fixed best_model
        st.code("Testing Prediction Function with Fixed Model:")


        if 'best_model' in locals():
            # Update the prediction function to use our fixed variables
            def predict_malaria_fixed(symptoms_dict):
                """
                Predict malaria based on symptoms

                Parameters:
                - symptoms_dict: dictionary with symptom values

                Example:
                {
                    'Age': 25,
                    'Gender': 1,  # 1 for Male, 0 for Female (after encoding)
                    'Fever': 1,
                    'Headache': 1,
                    'Chills': 1,
                    'Sweating': 0,
                    'Fatigue': 1,
                    'Nausea': 0,
                    'Vomiting': 0,
                    'Muscle_Pain': 1,
                    'Diarrhea': 0
                }
                """
                # Get global variables
                global best_model, scaler, X

                # Create a dataframe with all features
                features_df = pd.DataFrame(columns=X.columns)

                # Initialize with zeros for all features
                for col in X.columns:
                    features_df.loc[0, col] = 0

                # Update with provided symptoms
                missing_features = []
                for symptom, value in symptoms_dict.items():
                    if symptom in features_df.columns:
                        features_df.loc[0, symptom] = value
                    else:
                        missing_features.append(symptom)

                if missing_features:
                    st.code(f"Warning: These features were not in the model: {missing_features}")
                    st.code(f"Available features: {list(X.columns)}")

                # Check if all features are provided
                if len(symptoms_dict) < len(X.columns):
                    st.code(f"\nNote: Only {len(symptoms_dict)} features provided out of {len(X.columns)} expected.")
                    st.code("Missing features will be set to 0 (default).")

                # Scale the features
                features_scaled = scaler.transform(features_df)

                # Make prediction
                prediction = best_model.predict(features_scaled)[0]

                # Get probability if available
                probability = None
                if hasattr(best_model, 'predict_proba'):
                    probability = best_model.predict_proba(features_scaled)[0]

                best_model_name = st.session_state.best_model_name
                # Prepare result
                result = {
                    'prediction': int(prediction),
                    'prediction_label': 'Malaria Positive' if prediction == 1 else 'Malaria Negative',
                    'probability': probability,
                    'probability_positive': probability[1] if probability is not None else None,
                    'probability_negative': probability[0] if probability is not None else None,
                    'features_used': list(symptoms_dict.keys()),
                    'missing_features': [col for col in X.columns if col not in symptoms_dict],
                    'model_used': best_model_name
                }

                return result

            # Test with a complete example
            st.code("\nTest 1: Complete example with all features")
            complete_example = {
                'Age': 30,
                'Gender': 1,  # Assuming 1=Male, 0=Female (check your encoding)
                'Fever': 1,
                'Headache': 1,
                'Chills': 1,
                'Sweating': 0,
                'Fatigue': 1,
                'Nausea': 0,
                'Vomiting': 0,
                'Muscle_Pain': 1,
                'Diarrhea': 0
            }

            result = predict_malaria_fixed(complete_example)

            st.code(f"Model used: {result['model_used']}")
            st.code(f"Prediction: {result['prediction_label']}")
            if result['probability_positive'] is not None:
                st.code(f"Probability of Malaria: {result['probability_positive']:.2%}")
                st.code(f"Probability of No Malaria: {result['probability_negative']:.2%}")

            # Test with partial features
            st.code("\n" + "=" * 60)
            st.code("Test 2: Partial example (only key symptoms)")
            partial_example = {
                'Age': 25,
                'Gender': 0,  # Female
                'Fever': 1,
                'Headache': 1,
                'Chills': 1,
                'Fatigue': 1
            }

            result2 = predict_malaria_fixed(partial_example)
            st.code(f"\nPrediction: {result2['prediction_label']}")
            if result2['probability_positive'] is not None:
                st.code(f"Probability of Malaria: {result2['probability_positive']:.2%}")
            st.code(f"Missing features (set to 0): {result2['missing_features']}")

            # Create a quick demo function

            st.code("Quick Prediction Demo:")

            def quick_predict(age, gender, fever, headache, chills):
                """Quick prediction with common symptoms"""
                symptoms = {
                    'Age': age,
                    'Gender': gender,
                    'Fever': fever,
                    'Headache': headache,
                    'Chills': chills,
                    'Sweating': 0,  # Default values for others
                    'Fatigue': 0,
                    'Nausea': 0,
                    'Vomiting': 0,
                    'Muscle_Pain': 0,
                    'Diarrhea': 0
                }

                # Update with provided values
                if fever == 1:
                    symptoms['Fever'] = 1
                if headache == 1:
                    symptoms['Headache'] = 1
                if chills == 1:
                    symptoms['Chills'] = 1

                result = predict_malaria_fixed(symptoms)
                return result

            # Example usage
            st.code("\nExample 1: 25-year-old Male with Fever, Headache, and Chills")
            demo1 = quick_predict(age=25, gender=1, fever=1, headache=1, chills=1)
            st.code(f"Result: {demo1['prediction_label']}, Probability: {demo1['probability_positive']:.2%}")

            st.code("\nExample 2: 40-year-old Female with only Headache")
            demo2 = quick_predict(age=40, gender=0, fever=0, headache=1, chills=0)
            st.code(f"Result: {demo2['prediction_label']}, Probability: {demo2['probability_positive']:.2%}")

        else:
            st.code("best_model not found. Please run previous cells first.")



    def HyperparameterTuningforBestModel():
        st.write("### 19 Hyperparameter Tuning for Best Model")

        st.code (""" print("Hyperparameter Tuning for Best Model")
print("=" * 60)

if 'best_model_name' in locals() and best_model_name in models:
    from sklearn.model_selection import GridSearchCV
    
    print(f"Tuning hyperparameters for: {best_model_name}")
    
    # Define parameter grids based on model type
    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        },
        'Decision Tree': {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        },
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        },
        'K-Nearest Neighbors': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5]
        }
    }
    
    # Get appropriate parameter grid
    if best_model_name in param_grids:
        param_grid = param_grids[best_model_name]
        
        # Create base model
        base_model = models[best_model_name].__class__
        
        # Setup GridSearchCV
        grid_search = GridSearchCV(
            estimator=base_model(random_state=42),
            param_grid=param_grid,
            cv=5,  # 5-fold cross-validation
            scoring='accuracy',
            n_jobs=-1,  # Use all available cores
            verbose=1
        )
        
        print(f"Performing grid search with {len(param_grid)} parameter combinations...")
        
        # Fit grid search
        grid_search.fit(X_train_scaled_df, y_train)
        
        print("\nGrid Search Results:")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")
        
        # Evaluate on test set
        tuned_model = grid_search.best_estimator_
        y_pred_tuned = tuned_model.predict(X_test_scaled_df)
        tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
        
        print(f"Test Accuracy (Tuned): {tuned_accuracy:.4f}")
        
        # Compare with original
        original_accuracy = results[best_model_name]['test_accuracy']
        improvement = tuned_accuracy - original_accuracy
        
        print(f"\nComparison with Original Model:")
        print(f"Original Accuracy: {original_accuracy:.4f}")
        print(f"Tuned Accuracy: {tuned_accuracy:.4f}")
        print(f"Improvement: {improvement:.4f} ({improvement*100:.2f}%)")
        
        # Update best model if improved
        if improvement > 0:
            print("\nUpdating best model with tuned parameters...")
            best_model = tuned_model
            best_model_name = f"{best_model_name} (Tuned)"
            
            # Save the tuned model
            tuned_model_filename = 'saved_models/malaria_tuned_model.pkl'
            joblib.dump(tuned_model, tuned_model_filename)
            print(f"Tuned model saved as: {tuned_model_filename}")
        
        # Plot parameter performance
        if hasattr(grid_search, 'cv_results_'):
            results_df = pd.DataFrame(grid_search.cv_results_)
            
            # Get top 10 parameter combinations
            top_results = results_df.nsmallest(10, 'rank_test_score')
            
            plt.figure(figsize=(12, 6))
            plt.barh(range(len(top_results)), top_results['mean_test_score'])
            plt.yticks(range(len(top_results)), [f"Params {i+1}" for i in range(len(top_results))])
            plt.xlabel('Mean CV Accuracy')
            plt.title(f'Top 10 Parameter Combinations for {best_model_name}')
            plt.tight_layout()
            plt.show()
            
    else:
        print(f"No parameter grid defined for {best_model_name}")
        
else:
    print("Best model not found for tuning.") """)

        st.code("Hyperparameter Tuning for Best Model")

        best_model_name = st.session_state.best_model_name
        models = st.session_state.models
        if 'best_model_name' in locals() and best_model_name in models:
            from sklearn.model_selection import GridSearchCV

            st.code(f"Tuning hyperparameters for: {best_model_name}")

            # Define parameter grids based on model type
            param_grids = {
                'Random Forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'Logistic Regression': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                },
                'Decision Tree': {
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy']
                },
                'SVM': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto']
                },
                'K-Nearest Neighbors': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },
                'Gradient Boosting': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5]
                }
            }

            # Get appropriate parameter grid
            if best_model_name in param_grids:
                param_grid = param_grids[best_model_name]

                # Create base model
                base_model = models[best_model_name].__class__

                # Setup GridSearchCV
                grid_search = GridSearchCV(
                    estimator=base_model(random_state=42),
                    param_grid=param_grid,
                    cv=5,  # 5-fold cross-validation
                    scoring='accuracy',
                    n_jobs=-1,  # Use all available cores
                    verbose=1
                )

                st.code(f"Performing grid search with {len(param_grid)} parameter combinations...")

                # Fit grid search

                X_train_scaled_df = st.session_state.X_train_scaled_df
                y_train = st.session_state.y_train
                X_test_scaled_df = st.session_state.X_test_scaled_df
                y_test = st.session_state.y_test

                results = st.session_state.results

                grid_search.fit(X_train_scaled_df, y_train)

                st.code("\nGrid Search Results:")
                st.code(f"Best Parameters: {grid_search.best_params_}")
                st.code(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

                # Evaluate on test set
                tuned_model = grid_search.best_estimator_
                y_pred_tuned = tuned_model.predict(X_test_scaled_df)
                tuned_accuracy = accuracy_score(y_test, y_pred_tuned)

                st.code(f"Test Accuracy (Tuned): {tuned_accuracy:.4f}")

                # Compare with original
                original_accuracy = results[best_model_name]['test_accuracy']
                improvement = tuned_accuracy - original_accuracy

                st.code(f"\nComparison with Original Model:")
                st.code(f"Original Accuracy: {original_accuracy:.4f}")
                st.code(f"Tuned Accuracy: {tuned_accuracy:.4f}")
                st.code(f"Improvement: {improvement:.4f} ({improvement * 100:.2f}%)")

                # Update best model if improved
                if improvement > 0:
                    st.code("\nUpdating best model with tuned parameters...")
                    best_model = tuned_model
                    best_model_name = f"{best_model_name} (Tuned)"

                    # Save the tuned model
                    tuned_model_filename = 'saved_models/malaria_tuned_model.pkl'
                    joblib.dump(tuned_model, tuned_model_filename)
                    st.code(f"Tuned model saved as: {tuned_model_filename}")

                # Plot parameter performance
                if hasattr(grid_search, 'cv_results_'):
                    results_df = pd.DataFrame(grid_search.cv_results_)

                    # Get top 10 parameter combinations
                    top_results = results_df.nsmallest(10, 'rank_test_score')

                    plt.figure(figsize=(12, 6))
                    plt.barh(range(len(top_results)), top_results['mean_test_score'])
                    plt.yticks(range(len(top_results)), [f"Params {i + 1}" for i in range(len(top_results))])
                    plt.xlabel('Mean CV Accuracy')
                    plt.title(f'Top 10 Parameter Combinations for {best_model_name}')
                    plt.tight_layout()
                    st.pyplot(plt.gcf())

            else:
                st.code(f"No parameter grid defined for {best_model_name}")

        else:
            st.code("Best model not found for tuning.")


    def FinalSummaryModelDeploymentPreparation():
        st.write("### 20 Final Summary and Model Deployment Preparation")
        st.code("""print("FINAL MODEL SUMMARY")
print("=" * 60)

# Display final model information
if 'best_model' in locals():
    print(f"Best Model: {best_model_name}")
    print(f"Model Type: {type(best_model).__name__}")
    
    # Final evaluation
    y_pred_final = best_model.predict(X_test_scaled_df)
    final_accuracy = accuracy_score(y_test, y_pred_final)
    final_cm = confusion_matrix(y_test, y_pred_final)
    
    print(f"\nFinal Model Performance:")
    print(f"Test Accuracy: {final_accuracy:.4f}")
    print(f"Confusion Matrix:\n{final_cm}")
    
    # Calculate key metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    precision = precision_score(y_test, y_pred_final)
    recall = recall_score(y_test, y_pred_final)
    f1 = f1_score(y_test, y_pred_final)
    
    print(f"\nDetailed Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # ROC-AUC if probabilities available
    if hasattr(best_model, 'predict_proba'):
        y_pred_proba = best_model.predict_proba(X_test_scaled_df)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    # Feature importance summary
    print(f"\nTop 5 Most Important Features:")
    
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        for i, row in feature_importance.head().iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    elif hasattr(best_model, 'coef_'):
        coefficients = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': best_model.coef_[0],
            'Abs_Coefficient': np.abs(best_model.coef_[0])
        }).sort_values('Abs_Coefficient', ascending=False)
        
        for i, row in coefficients.head().iterrows():
            print(f"  {row['Feature']}: {row['Coefficient']:.4f}")
    
    print(f"\nModel Deployment Files:")
    print("✓ malaria_best_model.pkl - Trained model")
    print("✓ malaria_scaler.pkl - Feature scaler")
    print("✓ feature_names.pkl - Feature names")
    print("✓ all_malaria_models.pkl - All trained models")
    print("✓ model_results.pkl - Model performance results")
    
    print(f"\nNext Steps for Deployment:")
    print("1. Use predict_malaria_fixed() function for predictions")
    print("2. Load saved models with joblib.load()")
    print("3. Create a web API using Flask/FastAPI")
    print("4. Build a simple GUI with Tkinter or Streamlit")
    
    # Save final summary
    summary = {
        'best_model_name': best_model_name,
        'test_accuracy': final_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': final_cm.tolist(),
        'feature_names': X.columns.tolist(),
        'model_type': type(best_model).__name__
    }
    
    import json
    with open('saved_models/model_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: saved_models/model_summary.json")
    
else:
    print("Model not available for summary.")  """)

        st.code("FINAL MODEL SUMMARY")

        best_model_name = st.session_state.best_model_name
        X_test_scaled_df = st.session_state.X_test_scaled_df
        y_test =st.session_state.y_test
        X = st.session_state.X

        best_model = st.session_state.best_model

        # Display final model information
        if 'best_model' in locals():
            st.code(f"Best Model: {best_model_name}")
            st.code(f"Model Type: {type(best_model).__name__}")

            # Final evaluation
            y_pred_final = best_model.predict(X_test_scaled_df)
            final_accuracy = accuracy_score(y_test, y_pred_final)
            final_cm = confusion_matrix(y_test, y_pred_final)

            st.code(f"\nFinal Model Performance:")
            st.code(f"Test Accuracy: {final_accuracy:.4f}")
            st.code(f"Confusion Matrix:\n{final_cm}")

            # Calculate key metrics
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

            precision = precision_score(y_test, y_pred_final)
            recall = recall_score(y_test, y_pred_final)
            f1 = f1_score(y_test, y_pred_final)

            st.code(f"\nDetailed Metrics:")
            st.code(f"Precision: {precision:.4f}")
            st.code(f"Recall: {recall:.4f}")
            st.code(f"F1-Score: {f1:.4f}")

            # ROC-AUC if probabilities available
            if hasattr(best_model, 'predict_proba'):
                y_pred_proba = best_model.predict_proba(X_test_scaled_df)[:, 1]
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                st.code(f"ROC-AUC Score: {roc_auc:.4f}")

            # Feature importance summary
            st.code(f"\nTop 5 Most Important Features:")

            if hasattr(best_model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': best_model.feature_importances_
                }).sort_values('Importance', ascending=False)

                for i, row in feature_importance.head().iterrows():
                    print(f"  {row['Feature']}: {row['Importance']:.4f}")

            elif hasattr(best_model, 'coef_'):
                coefficients = pd.DataFrame({
                    'Feature': X.columns,
                    'Coefficient': best_model.coef_[0],
                    'Abs_Coefficient': np.abs(best_model.coef_[0])
                }).sort_values('Abs_Coefficient', ascending=False)

                for i, row in coefficients.head().iterrows():
                    print(f"  {row['Feature']}: {row['Coefficient']:.4f}")

            st.code(f"\nModel Deployment Files:")
            st.code("✓ malaria_best_model.pkl - Trained model")
            st.code("✓ malaria_scaler.pkl - Feature scaler")
            st.code("✓ feature_names.pkl - Feature names")
            st.code("✓ all_malaria_models.pkl - All trained models")
            st.code("✓ model_results.pkl - Model performance results")

            st.code(f"\nNext Steps for Deployment:")
            st.code("1. Use predict_malaria_fixed() function for predictions")
            st.code("2. Load saved models with joblib.load()")
            st.code("3. Create a web API using Flask/FastAPI")
            st.code("4. Build a simple GUI with Tkinter or Streamlit")

            # Save final summary
            summary = {
                'best_model_name': best_model_name,
                'test_accuracy': final_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': final_cm.tolist(),
                'feature_names': X.columns.tolist(),
                'model_type': type(best_model).__name__
            }

            import json
            with open('saved_models/model_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)

            st.code(f"\nSummary saved to: saved_models/model_summary.json")

        else:
            st.code("Model not available for summary.")



    if __name__ == "__main__":
        library()
        loadDataset()
        CheckDataTypes()
        HandleCategoricalVariables()
        CheckforMissingValues()
        HandelingMissingValues()
        CheckTargetVariableDistribution()
        NowTryCorrelationAnalysis()
        exploreFeatureDistributions()
        SplitFeaturesAndTarget()
        SplitDataTrainingTestingSets()
        FeatureScaling()
        TrainMultipleModels()
        CompareModelPerformance()
        DetailedEvaluationofBestModel()
        FeatureImportanceAnalysis()
        SaveTheBestModels()
        predictionfunction()
        HyperparameterTuningforBestModel()
        FinalSummaryModelDeploymentPreparation()













# About Page
elif choice == "ℹ️ About":
    # Inject custom CSS for styling
    st.markdown("""
    <style>
        .about-container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .hero-section { background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 20px; padding: 40px; color: white; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.1); margin-bottom:30px;}
        .hero-title { font-size: 2.8rem; font-weight: 800; margin-bottom: 15px; font-family: 'Segoe UI', system-ui; }
        .hero-subtitle { font-size: 1.2rem; opacity: 0.9; max-width: 800px; margin:0 auto; line-height:1.6; }
        .info-card { background: white; border-radius: 15px; padding: 25px; box-shadow: 0 5px 15px rgba(0,0,0,0.08); border:1px solid #eaeaea; transition: all 0.3s ease; margin-bottom:20px;}
        .info-card:hover { transform: translateY(-5px); box-shadow: 0 8px 25px rgba(0,0,0,0.12);}
        .card-title { color:#2c3e50; font-size:1.4rem; font-weight:700; margin-bottom:15px; display:flex; align-items:center; gap:10px; }
        .card-content { color:#546e7a; line-height:1.7; font-size:1rem; }
        .model-cards { display:flex; flex-wrap:wrap; gap:10px; margin-top:10px; }
        .model-tag { background: linear-gradient(135deg, #667eea, #764ba2); color:white; padding:6px 14px; border-radius:20px; font-size:0.85rem; font-weight:500; white-space:nowrap; }
        .developer-profile { background: linear-gradient(135deg, #f5f7fa, #c3cfe2); border-radius:15px; padding:25px; margin:30px 0; }
        .profile-header { display:flex; align-items:center; gap:20px; margin-bottom:20px; }
        .avatar-placeholder { width:80px; height:80px; background: linear-gradient(135deg, #667eea, #764ba2); border-radius:50%; display:flex; align-items:center; justify-content:center; color:white; font-size:2rem; font-weight:bold; }
        .profile-info h3 { margin:0; color:#2c3e50; font-size:1.5rem; }
        .profile-info p { margin:5px 0 0 0; color:#7f8c8d; }
        .contact-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:15px; margin-top:20px; }
        .contact-item { background:white; padding:15px; border-radius:10px; text-align:center; transition: transform 0.2s; }
        .contact-item:hover { transform: scale(1.05); }
        .contact-icon { font-size:1.8rem; margin-bottom:10px; display:block; }
        .warning-banner { background: linear-gradient(135deg,#ff6b6b,#ee5a52); color:white; padding:20px; border-radius:15px; margin:30px 0; text-align:center; }
        .warning-icon { font-size:2rem; margin-bottom:15px; }
        .page-footer { text-align:center; color:#7f8c8d; margin-top:40px; padding-top:20px; border-top:1px solid #ecf0f1; font-size:0.9rem; }
        @media(max-width:768px){ .hero-title{font-size:2rem;} .info-grid{grid-template-columns:1fr;} .profile-header{flex-direction:column;text-align:center;} }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="about-container">', unsafe_allow_html=True)

    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🏥 Malaria Prediction System</h1>
        <p class="hero-subtitle">
            An AI-powered diagnostic tool to assist healthcare professionals 
            in preliminary malaria risk assessment using advanced machine learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2,1], gap="large")

    with col1:
        # Purpose
        st.markdown("""
        <div class="info-card">
            <div class="card-title">🎯 Purpose & Mission</div>
            <div class="card-content">
                Provides preliminary assessment of malaria infection risk based on symptoms and epidemiological factors. 
                Supports early detection and emphasizes professional medical consultation and laboratory confirmation.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Technology Stack
        st.markdown("""
        <div class="info-card">
            <div class="card-title">🛠️ Technology Stack</div>
            <div class="card-content">
                <ul>
                    <li>🤖 Machine Learning: Scikit-learn, TensorFlow</li>
                    <li>🌐 Web Framework: Streamlit, FastAPI</li>
                    <li>📊 Data Processing: Pandas, NumPy</li>
                    <li>📈 Visualization: Plotly, Matplotlib</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Model Architecture
        st.markdown("""
        <div class="info-card">
            <div class="card-title">🧠 Model Architecture</div>
            <div class="card-content">
                Ensemble learning combining Random Forest, Gradient Boosting, Logistic Regression, Neural Networks, SVM.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Disclaimer
        st.markdown("""
        <div class="warning-banner">
            <div class="warning-icon">⚠️</div>
            <p>This tool is for informational purposes only and <strong>NOT</strong> a substitute for professional medical advice. Always consult healthcare professionals. Malaria requires lab confirmation.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Developer Info
        st.markdown("""
        <div class="developer-profile">
            <div class="profile-header">
                <div class="avatar-placeholder">KIK</div>
                <div class="profile-info">
                    <h3>Khairullah Ibrahim Khail</h3>
                    <p>Lead Developer & Data Scientist</p>
                </div>
            </div>
            <div class="contact-grid">
                <div class="contact-item">
                    <span class="contact-icon">📧</span>Email<br><small><a href="mailto:ibrahimkhil975@gmail.com">ibrahimkhil975@gmail.com</a></small>
                </div>
                <div class="contact-item">
                    <span class="contact-icon">📱</span>WhatsApp<br><small>+977 88770458</small>
                </div>
                <div class="contact-item">
                    <span class="contact-icon">🚀</span>Version<br><small>2.0.0 (2025)</small>
                </div>
                <div class="contact-item">
                    <span class="contact-icon">🔒</span>Privacy<br><small>HIPAA Compliant</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # References
        st.markdown("""
        <div class="info-card">
            <div class="card-title">📚 References</div>
            <div class="card-content">
                <ul>
                    <li>WHO Malaria Guidelines 2024</li>
                    <li>CDC Clinical Protocols</li>
                    <li>Lancet Medical Journal</li>
                    <li>Clinical Trial Data</li>
                    <li>ML Healthcare Research</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="page-footer">
        © 2025 Malaria Prediction System | Developed for educational and research purposes.<br>
        Always consult healthcare professionals for medical decisions.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Run instructions
st.sidebar.markdown("---")
with st.sidebar.expander("How to Run", icon="💡"):
    st.markdown("""
    1. **Install requirements:**
    ```bash
    pip install streamlit pandas numpy scikit-learn matplotlib seaborn plotly pillow joblib
    ```
    2. **Run the app:** 
    ```bash
    streamlit run malaria_app.py
    ```
    3. **Load models** from Home page.
    4. **Start predicting!**
    """)




jupyterlab_pygments~=0.3.0
pip~=25.3
python-box~=7.3.2
configparser~=7.2.0
attrs~=23.2.0
distro~=1.9.0
PySocks~=1.7.1
protobuf~=4.21.12
pyOpenSSL~=25.1.0
cryptography~=43.0.0
filelock~=3.14.0
redis~=4.3.4
streamlit~=1.52.1
pandas~=2.3.3
numpy~=2.2.4
joblib~=1.5.2
matplotlib~=3.10.8
seaborn~=0.13.2
plotly~=6.5.0
scikit-learn~=1.8.0
pillow~=11.1.0
nbformat~=5.9.1
nbconvert~=7.16.6
ipython~=8.20.0