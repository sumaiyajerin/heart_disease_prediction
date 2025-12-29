import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Professional Colors
st.markdown("""
    <style>
    /* Main theme colors - Customizable */
    :root {
        --primary-red: #e74c3c;
        --dark-red: #c0392b;
        --primary-blue: #3498db;
        --success-green: #27ae60;
        --warning-orange: #f39c12;
        --dark-bg: #0f3460;
        --card-bg: #16213e;
        --light-text: #ecf0f1;
        --gray-text: #95a5a6;
    }
    
    .main { 
        padding: 2rem; 
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(231,76,60,0.3);
    }
    
    .stButton > button:hover {
        box-shadow: 0 8px 25px rgba(231,76,60,0.5);
        transform: translateY(-2px);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    h1, h2, h3, p, span, label {
        color: #ecf0f1;
    }
    
    .success-box {
        background: linear-gradient(135deg, #27ae60 0%, #229954 100%);
        border-left: 5px solid #1abc9c;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(39,174,96,0.3);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        border-left: 5px solid #d68910;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(243,156,18,0.3);
    }
    
    .error-box {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        border-left: 5px solid #a93226;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(231,76,60,0.3);
    }
    
    .footer {
        text-align: center;
        padding: 20px;
        color: #95a5a6;
        border-top: 1px solid #334155;
        margin-top: 40px;
        font-size: 0.85em;
    }
    </style>
    """, unsafe_allow_html=True)

# Language Dictionary
LANGUAGES = {
    "English": {
        "title": "❤️ Heart Disease Prediction System",
        "subtitle": "Interactive Machine Learning Diagnosis Tool",
        "nav_patient": "🏥 Patient Prediction",
        "nav_model": "📊 Model Comparison",
        "nav_data": "📈 Data Analysis",
        "nav_about": "ℹ️ About",
        
        "lang_select": "Select Language / ভাষা নির্বাচন করুন",
        "english": "English",
        "bengali": "বাংলা",
        
        # Patient Prediction Page
        "patient_title": "Patient Risk Assessment",
        "patient_subtitle": "Enter medical parameters for instant prediction",
        "personal_info": "Personal Information",
        "age": "Age (years)",
        "gender": "Gender",
        "male": "Male",
        "female": "Female",
        "cardiac_info": "Cardiac Information",
        "chest_pain": "Chest Pain Type",
        "typical_angina": "Typical Angina",
        "atypical_angina": "Atypical Angina",
        "non_anginal": "Non-anginal",
        "asymptomatic": "Asymptomatic",
        "blood_params": "Blood Parameters",
        "resting_bp": "Resting BP (mmHg)",
        "cholesterol": "Cholesterol (mg/dl)",
        "fasting_bs": "Fasting BS > 120",
        "no": "No",
        "yes": "Yes",
        "ecg_exercise": "ECG & Exercise",
        "resting_ecg": "Resting ECG",
        "normal": "Normal",
        "st_abnormality": "ST-T Abnormality",
        "lv_hypertrophy": "LV Hypertrophy",
        "max_heart_rate": "Max Heart Rate",
        "exercise_angina": "Exercise Angina",
        "st_analysis": "ST Segment Analysis",
        "st_depression": "ST Depression",
        "st_slope": "ST Slope",
        "upsloping": "Upsloping",
        "flat": "Flat",
        "downsloping": "Downsloping",
        "additional": "Additional Factors",
        "major_vessels": "Major Vessels (0-3)",
        "thalassemia": "Thalassemia",
        "thal_normal": "Normal",
        "thal_fixed": "Fixed Defect",
        "thal_reversible": "Reversible",
        "predict_btn": "🔍 PREDICT HEART DISEASE RISK",
        "analyzing": "🔄 Analyzing patient data...",
        "results_title": "Prediction Results",
        "risk_level": "Risk Level",
        "disease_probability": "Disease Probability",
        "models_agree": "Model Consensus",
        "low_risk": "🟢 LOW RISK",
        "moderate_risk": "🟡 MODERATE RISK",
        "high_risk": "🔴 HIGH RISK",
        "individual_pred": "🤖 Individual Model Predictions",
        "model": "Model",
        "prediction": "Prediction",
        "probability": "Risk %",
        "no_disease": "❌ No Disease",
        "disease": "⚠️ Disease",
        "visualization": "📊 Risk Analysis Visualization",
        "model_risk": "Model Risk Prediction",
        "model_consensus": "Model Consensus",
        "clinical_rec": "🏥 Clinical Recommendation",
        "high_risk_msg": "HIGH RISK - Seek immediate medical consultation!",
        "moderate_risk_msg": "MODERATE RISK - Schedule a medical checkup soon",
        "low_risk_msg": "LOW RISK - Continue regular health monitoring",
        "patient_summary": "📋 Patient Data Summary",
        "parameter": "Parameter",
        "value": "Value",
        
        # Model Comparison
        "model_perf": "Model Performance",
        "model_perf_sub": "Compare all machine learning models",
        "perf_metrics": "Performance Metrics",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1-Score",
        "roc_auc": "ROC-AUC",
        
        # Data Analysis
        "data_analysis": "Dataset Exploration",
        "data_sub": "Comprehensive data analysis and statistics",
        "total_samples": "Total Samples",
        "total_features": "Total Features",
        "disease_cases": "Disease Cases",
        "healthy_cases": "Healthy Cases",
        "dataset_preview": "Dataset Preview",
        "statistics": "Statistics",
        "disease_dist": "Disease Distribution",
        "age_dist": "Age Distribution",
        
        # About
        "about_title": "About This Application",
        "about_sub": "Learn about the project and its capabilities",
        "overview": "Project Overview",
        "overview_text": "This is a comprehensive machine learning application for predicting heart disease risk based on medical parameters.",
        "ml_models": "Machine Learning Models",
        "model1": "Logistic Regression - Linear classification model",
        "model2": "Decision Tree - Interpretable tree-based model",
        "model3": "Random Forest - Ensemble method (Best ROC-AUC)",
        "model4": "Support Vector Machine - Advanced kernel-based classifier",
        "model5": "K-Nearest Neighbors - Instance-based learner",
        "dataset_info": "Dataset Information",
        "samples": "Total Samples",
        "features": "Features",
        "perf_results": "Model Performance",
        "best_acc": "Best Accuracy",
        "best_roc": "Best ROC-AUC",
        "best_recall": "Best Recall",
        "disclaimer": "Important Disclaimer",
        "disclaimer_text": "This application is for educational and informational purposes only. It is NOT a substitute for professional medical diagnosis.",
        "privacy": "Privacy & Security",
        "privacy_text": "No data is stored, no transmission, all local computation",
        "tech_used": "Technologies Used",
    },
    
    "বাংলা": {
        "title": "❤️ হৃদরোগ পূর্বাভাস ব্যবস্থা",
        "subtitle": "ইন্টারঅ্যাক্টিভ মেশিন লার্নিং ডায়াগনসিস টুল",
        "nav_patient": "🏥 রোগী পূর্বাভাস",
        "nav_model": "📊 মডেল তুলনা",
        "nav_data": "📈 ডেটা বিশ্লেষণ",
        "nav_about": "ℹ️ সম্পর্কে",
        
        "lang_select": "ভাষা নির্বাচন করুন / Select Language",
        "english": "English",
        "bengali": "বাংলা",
        
        # Patient Prediction Page
        "patient_title": "রোগীর ঝুঁকি মূল্যায়ন",
        "patient_subtitle": "তাৎক্ষণিক পূর্বাবাসের জন্য চিকিৎসা পরামিতি প্রবেশ করুন",
        "personal_info": "ব্যক্তিগত তথ্য",
        "age": "বয়স (বছর)",
        "gender": "লিঙ্গ",
        "male": "পুরুষ",
        "female": "মহিলা",
        "cardiac_info": "কার্ডিয়াক তথ্য",
        "chest_pain": "বুকের ব্যথার ধরন",
        "typical_angina": "সাধারণ এনজাইনা",
        "atypical_angina": "অস্বাভাবিক এনজাইনা",
        "non_anginal": "অ-এনজাইনা",
        "asymptomatic": "উপসর্গবিহীন",
        "blood_params": "রক্ত পরামিতি",
        "resting_bp": "বিশ্রামের রক্তচাপ (mmHg)",
        "cholesterol": "সিরাম কোলেস্টেরল (mg/dl)",
        "fasting_bs": "উপবাসের রক্ত শর্করা > 120",
        "no": "না",
        "yes": "হ্যাঁ",
        "ecg_exercise": "ইসিজি এবং ব্যায়াম",
        "resting_ecg": "বিশ্রামের ইসিজি",
        "normal": "স্বাভাবিক",
        "st_abnormality": "এসটি-টি অস্বাভাবিকতা",
        "lv_hypertrophy": "এলভি হাইপারট্রফি",
        "max_heart_rate": "সর্বোচ্চ হৃদস্পন্দন",
        "exercise_angina": "ব্যায়াম-প্রবর্তিত এনজাইনা",
        "st_analysis": "এসটি সেগমেন্ট বিশ্লেষণ",
        "st_depression": "এসটি বিষণ্নতা",
        "st_slope": "এসটি ঢাল",
        "upsloping": "উর্ধ্বমুখী",
        "flat": "সমতল",
        "downsloping": "নিম্নমুখী",
        "additional": "অতিরিক্ত কারণ",
        "major_vessels": "প্রধান রক্তনালি (0-3)",
        "thalassemia": "থ্যালাসেমিয়া",
        "thal_normal": "স্বাভাবিক",
        "thal_fixed": "নির্ধারিত ত্রুটি",
        "thal_reversible": "উল্টানো যায় এমন",
        "predict_btn": "🔍 হৃদরোগের ঝুঁকি পূর্বাভাস করুন",
        "analyzing": "🔄 রোগীর ডেটা বিশ্লেষণ করছি...",
        "results_title": "পূর্বাভাস ফলাফল",
        "risk_level": "ঝুঁকির স্তর",
        "disease_probability": "রোগের সম্ভাবনা",
        "models_agree": "মডেল সর্বসম্মতি",
        "low_risk": "🟢 কম ঝুঁকি",
        "moderate_risk": "🟡 মধ্যম ঝুঁকি",
        "high_risk": "🔴 উচ্চ ঝুঁকি",
        "individual_pred": "🤖 ব্যক্তিগত মডেল পূর্বাভাস",
        "model": "মডেল",
        "prediction": "পূর্বাভাস",
        "probability": "ঝুঁকি %",
        "no_disease": "❌ রোগ নেই",
        "disease": "⚠️ রোগ",
        "visualization": "📊 ঝুঁকি বিশ্লেষণ ভিজুয়ালাইজেশন",
        "model_risk": "মডেল ঝুঁকি পূর্বাভাস",
        "model_consensus": "মডেল সর্বসম্মতি",
        "clinical_rec": "🏥 ক্লিনিকাল সুপারিশ",
        "high_risk_msg": "উচ্চ ঝুঁকি - তাৎক্ষণিক চিকিৎসা পরামর্শ নিন!",
        "moderate_risk_msg": "মধ্যম ঝুঁকি - শীঘ্রই চিকিৎসা পরীক্ষা নির্ধারণ করুন",
        "low_risk_msg": "কম ঝুঁকি - নিয়মিত স্বাস্থ্য পর্যবেক্ষণ চালিয়ে যান",
        "patient_summary": "📋 রোগীর ডেটা সারসংক্ষেপ",
        "parameter": "প্যারামিটার",
        "value": "মান",
        
        # Model Comparison
        "model_perf": "মডেল পারফরম্যান্স",
        "model_perf_sub": "সমস্ত মেশিন লার্নিং মডেলের তুলনা করুন",
        "perf_metrics": "পারফরম্যান্স মেট্রিক্স",
        "accuracy": "নির্ভুলতা",
        "precision": "নির্ভুলতা",
        "recall": "রিকল",
        "f1": "F1-স্কোর",
        "roc_auc": "ROC-AUC",
        
        # Data Analysis
        "data_analysis": "ডেটাসেট অন্বেষণ",
        "data_sub": "ব্যাপক ডেটা বিশ্লেষণ এবং পরিসংখ্যান",
        "total_samples": "মোট নমুনা",
        "total_features": "মোট বৈশিষ্ট্য",
        "disease_cases": "রোগের কেস",
        "healthy_cases": "সুস্থ কেস",
        "dataset_preview": "ডেটাসেট প্রিভিউ",
        "statistics": "পরিসংখ্যান",
        "disease_dist": "রোগের বিতরণ",
        "age_dist": "বয়সের বিতরণ",
        
        # About
        "about_title": "এই অ্যাপ্লিকেশন সম্পর্কে",
        "about_sub": "প্রকল্প এবং এর ক্ষমতা সম্পর্কে জানুন",
        "overview": "প্রকল্প সংক্ষিপ্ত বিবরণ",
        "overview_text": "এটি হৃদরোগ ঝুঁকি পূর্বাবাসের জন্য চিকিৎসা পরামিতির উপর ভিত্তি করে একটি ব্যাপক মেশিন লার্নিং অ্যাপ্লিকেশন।",
        "ml_models": "মেশিন লার্নিং মডেল",
        "model1": "লজিস্টিক রিগ্রেশন - রৈখিক শ্রেণীবিভাগ মডেল",
        "model2": "সিদ্ধান্ত গাছ - ব্যাখ্যাযোগ্য গাছ-ভিত্তিক মডেল",
        "model3": "র‍্যান্ডম ফরেস্ট - এনসেম্বল পদ্ধতি (সর্বোত্তম ROC-AUC)",
        "model4": "সাপোর্ট ভেক্টর মেশিন - উন্নত কার্নেল-ভিত্তিক শ্রেণীবিভাগ",
        "model5": "কে-নিয়ারেস্ট নেইবার্স - উদাহরণ-ভিত্তিক শিক্ষা",
        "dataset_info": "ডেটাসেট তথ্য",
        "samples": "মোট নমুনা",
        "features": "বৈশিষ্ট্য",
        "perf_results": "মডেল কর্মক্ষমতা",
        "best_acc": "সর্বোত্তম নির্ভুলতা",
        "best_roc": "সর্বোত্তম ROC-AUC",
        "best_recall": "সর্বোত্তম রিকল",
        "disclaimer": "গুরুত্বপূর্ণ অস্বীকৃতি",
        "disclaimer_text": "এই অ্যাপ্লিকেশনটি শুধুমাত্র শিক্ষামূলক এবং তথ্যমূলক উদ্দেশ্যে। এটি পেশাদার চিকিৎসা নির্ণয়ের বিকল্প নয়।",
        "privacy": "গোপনীয়তা এবং নিরাপত্তা",
        "privacy_text": "কোনো ডেটা সংরক্ষিত নয়, কোনো প্রেরণ নেই, সমস্ত স্থানীয় গণনা",
        "tech_used": "ব্যবহৃত প্রযুক্তি",
    }
}

# Initialize session state
if 'language' not in st.session_state:
    st.session_state.language = "English"

# Load data with error handling
try:
    df = pd.read_csv('heart_disease.csv')
except FileNotFoundError:
    st.error("❌ Error: heart_disease.csv file not found!")
    st.info("Please make sure heart_disease.csv is in the same directory as this script.")
    st.stop()

# Language selector in sidebar
with st.sidebar:
    # Try to load and display logo
    try:
        from PIL import Image
        heart_img = Image.open('heart_icon.png')
        st.image(heart_img, width=200)
    except:
        st.markdown("❤️")
    
    st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); border-radius: 12px; margin-bottom: 30px;'>
            <h3 style='color: white; margin: 5px 0;'>Heart Predictor</h3>
            <p style='color: #ecf0f1; font-size: 0.9em; margin: 0;'>Jerin_Papri_Mithila</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Language Selection
    lang_choice = st.radio(
        LANGUAGES["English"]["lang_select"],
        ["English", "বাংলা"],
        label_visibility="collapsed"
    )
    st.session_state.language = lang_choice
    
    lang = LANGUAGES[st.session_state.language]
    
    st.markdown("---")
    st.markdown(f"### 🎯 Navigation")
    page = st.radio(
        "Select Page",
        [lang["nav_patient"], lang["nav_model"], lang["nav_data"], lang["nav_about"]],
        label_visibility="collapsed"
    )

# Get current language
lang = LANGUAGES[st.session_state.language]

# Main title
st.markdown(f"""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1 style='color: #e74c3c; margin: 0; font-size: 2.5em;'>{lang['title']}</h1>
        <p style='color: #95a5a6; font-size: 1.1em; margin: 10px 0;'>{lang['subtitle']}</p>
    </div>
""", unsafe_allow_html=True)

# ============ PAGE 1: PATIENT PREDICTION ============
if page == lang["nav_patient"]:
    st.markdown("---")
    st.markdown(f"### 📋 {lang['patient_subtitle']}")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown(f"### 👤 {lang['personal_info']}")
        age = st.slider(f"{lang['age']}", 30, 80, 50)
        sex = st.radio(f"{lang['gender']}", [lang['male'], lang['female']], horizontal=True)
        sex_val = 1 if sex == lang['male'] else 0
        
        st.markdown(f"### 💓 {lang['cardiac_info']}")
        cp_options = [
            (lang['typical_angina'], 1),
            (lang['atypical_angina'], 2),
            (lang['non_anginal'], 3),
            (lang['asymptomatic'], 4)
        ]
        cp = st.selectbox(lang['chest_pain'], cp_options, format_func=lambda x: x[0])
        cp_val = cp[1]
        
        st.markdown(f"### 🩸 {lang['blood_params']}")
        trestbps = st.slider(lang['resting_bp'], 80, 200, 120)
        chol = st.slider(lang['cholesterol'], 100, 400, 200)
        fbs = st.radio(lang['fasting_bs'], [lang['no'], lang['yes']], horizontal=True)
        fbs_val = 1 if fbs == lang['yes'] else 0
    
    with col2:
        st.markdown(f"### 📊 {lang['ecg_exercise']}")
        restecg_options = [(lang['normal'], 0), (lang['st_abnormality'], 1), (lang['lv_hypertrophy'], 2)]
        restecg = st.selectbox(lang['resting_ecg'], restecg_options, format_func=lambda x: x[0])
        restecg_val = restecg[1]
        thalach = st.slider(lang['max_heart_rate'], 60, 220, 150)
        exang = st.radio(lang['exercise_angina'], [lang['no'], lang['yes']], horizontal=True)
        exang_val = 1 if exang == lang['yes'] else 0
        
        st.markdown(f"### 📈 {lang['st_analysis']}")
        oldpeak = st.slider(lang['st_depression'], 0.0, 6.0, 1.0, step=0.1)
        slope_options = [(lang['upsloping'], 1), (lang['flat'], 2), (lang['downsloping'], 3)]
        slope = st.selectbox(lang['st_slope'], slope_options, format_func=lambda x: x[0])
        slope_val = slope[1]
        
        st.markdown(f"### 🔴 {lang['additional']}")
        ca = st.slider(lang['major_vessels'], 0, 4, 0)
        thal_options = [(lang['thal_normal'], 1), (lang['thal_fixed'], 2), (lang['thal_reversible'], 3)]
        thal = st.selectbox(lang['thalassemia'], thal_options, format_func=lambda x: x[0])
        thal_val = thal[1]
    
    st.markdown("---")
    
    col_predict = st.columns([1, 3, 1])
    with col_predict[1]:
        predict_button = st.button(lang['predict_btn'], use_container_width=True, key="predict")
    
    if predict_button:
        X = df.drop('target', axis=1)
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        user_data = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val, thalach, exang_val, oldpeak, slope_val, ca, thal_val]])
        user_data_scaled = scaler.transform(user_data)
        
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        
        predictions = {}
        probabilities = {}
        
        with st.spinner(lang['analyzing']):
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                pred = model.predict(user_data_scaled)[0]
                
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(user_data_scaled)[0][1] * 100
                else:
                    prob = 50
                
                predictions[name] = lang['no_disease'] if pred == 0 else lang['disease']
                probabilities[name] = prob
        
        st.markdown("---")
        st.markdown(f"### {lang['results_title']}")
        
        avg_prob = np.mean(list(probabilities.values()))
        
        if avg_prob < 30:
            risk = lang['low_risk']
        elif avg_prob < 60:
            risk = lang['moderate_risk']
        else:
            risk = lang['high_risk']
        
        metric_cols = st.columns(3, gap="large")
        
        with metric_cols[0]:
            st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size: 0.9em; color: #95a5a6; margin-bottom: 10px;'>{lang['risk_level']}</div>
                    <div style='font-size: 2em; color: #e74c3c; font-weight: bold;'>{risk}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with metric_cols[1]:
            st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size: 0.9em; color: #95a5a6; margin-bottom: 10px;'>{lang['disease_probability']}</div>
                    <div style='font-size: 2em; color: #3498db; font-weight: bold;'>{avg_prob:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
        
        with metric_cols[2]:
            models_agree = sum(1 for v in probabilities.values() if v > 50)
            st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size: 0.9em; color: #95a5a6; margin-bottom: 10px;'>{lang['models_agree']}</div>
                    <div style='font-size: 2em; color: #27ae60; font-weight: bold;'>{models_agree}/5</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown(f"### {lang['individual_pred']}")
        results_df = pd.DataFrame({
            lang['model']: list(predictions.keys()),
            lang['prediction']: list(predictions.values()),
            lang['probability']: [f"{v:.1f}%" for v in probabilities.values()]
        })
        
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown(f"### {lang['visualization']}")
        
        viz_col1, viz_col2 = st.columns(2, gap="large")
        
        with viz_col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#0f3460')
            ax.set_facecolor('#1a1a2e')
            
            colors = ['#e74c3c' if v > 50 else '#27ae60' for v in probabilities.values()]
            ax.barh(list(probabilities.keys()), list(probabilities.values()), color=colors, edgecolor='white', linewidth=1.5)
            ax.axvline(x=50, color='white', linestyle='--', linewidth=2, alpha=0.5)
            ax.set_xlabel(f"{lang['disease_probability']} (%)", color='white', fontsize=11, fontweight='bold')
            ax.set_title(lang['model_risk'], color='white', fontsize=13, fontweight='bold', pad=20)
            ax.set_xlim(0, 100)
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            st.pyplot(fig)
        
        with viz_col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#0f3460')
            ax.set_facecolor('#1a1a2e')
            
            risk_count = sum(1 for v in probabilities.values() if v > 50)
            no_risk_count = len(probabilities) - risk_count
            
            colors_pie = ['#e74c3c', '#27ae60']
            wedges, texts, autotexts = ax.pie(
                [risk_count, no_risk_count],
                labels=[lang['disease'], lang['no_disease']],
                autopct='%1.0f%%',
                colors=colors_pie,
                startangle=90,
                textprops={'color': 'white', 'fontweight': 'bold', 'fontsize': 11}
            )
            ax.set_title(lang['model_consensus'], color='white', fontsize=13, fontweight='bold', pad=20)
            
            st.pyplot(fig)
        
        st.markdown("---")
        st.markdown(f"### {lang['clinical_rec']}")
        
        if avg_prob > 70:
            st.markdown(f"""
                <div class='error-box'>
                    <h3 style='margin: 0; color: white;'>{lang['high_risk']}</h3>
                    <p style='margin: 10px 0 0 0; color: white;'>{lang['high_risk_msg']}</p>
                </div>
            """, unsafe_allow_html=True)
        elif avg_prob > 50:
            st.markdown(f"""
                <div class='warning-box'>
                    <h3 style='margin: 0; color: white;'>{lang['moderate_risk']}</h3>
                    <p style='margin: 10px 0 0 0; color: white;'>{lang['moderate_risk_msg']}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='success-box'>
                    <h3 style='margin: 0; color: white;'>{lang['low_risk']}</h3>
                    <p style='margin: 10px 0 0 0; color: white;'>{lang['low_risk_msg']}</p>
                </div>
            """, unsafe_allow_html=True)

elif page == lang["nav_model"]:
    st.markdown(f"### {lang['model_perf']}")
    st.markdown(f"{lang['model_perf_sub']}")
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test_scaled)
        
        results[name] = {
            lang['accuracy']: accuracy_score(y_test, y_pred),
            lang['precision']: precision_score(y_test, y_pred),
            lang['recall']: recall_score(y_test, y_pred),
            lang['f1']: f1_score(y_test, y_pred),
            lang['roc_auc']: roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
        }
    
    results_df = pd.DataFrame(results).T
    st.markdown(f"### {lang['perf_metrics']}")
    st.dataframe(results_df, use_container_width=True)

elif page == lang["nav_data"]:
    st.markdown(f"### {lang['data_analysis']}")
    st.markdown(f"{lang['data_sub']}")
    
    col1, col2, col3, col4 = st.columns(4, gap="small")
    
    with col1:
        st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size: 0.85em; color: #95a5a6;'>{lang['total_samples']}</div>
                <div style='font-size: 2em; color: #3498db; font-weight: bold;'>{len(df)}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size: 0.85em; color: #95a5a6;'>{lang['total_features']}</div>
                <div style='font-size: 2em; color: #27ae60; font-weight: bold;'>{len(df.columns)-1}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        disease_count = (df['target'] == 1).sum()
        st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size: 0.85em; color: #95a5a6;'>{lang['disease_cases']}</div>
                <div style='font-size: 2em; color: #e74c3c; font-weight: bold;'>{disease_count}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        healthy = (df['target'] == 0).sum()
        st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size: 0.85em; color: #95a5a6;'>{lang['healthy_cases']}</div>
                <div style='font-size: 2em; color: #27ae60; font-weight: bold;'>{healthy}</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown(f"### {lang['dataset_preview']}")
    st.dataframe(df.head(10), use_container_width=True, hide_index=False)

elif page == lang["nav_about"]:
    st.markdown(f"### {lang['about_title']}")
    st.markdown(f"{lang['about_sub']}")
    
    st.markdown(f"""
        #### {lang['overview']}
        {lang['overview_text']}
        
        #### {lang['ml_models']}
        1. {lang['model1']}
        2. {lang['model2']}
        3. {lang['model3']}
        4. {lang['model4']}
        5. {lang['model5']}
        
        #### {lang['disclaimer']}
        {lang['disclaimer_text']}
    """)

# FOOTER with Copyright
st.markdown("---")
st.markdown("""
    <div class='footer'>
        <p style='margin: 0; color: #95a5a6; font-size: 0.9em;'>
            ❤️ Heart Disease Prediction System<br>
            <span style='font-size: 0.85em;'>Advanced Machine Learning for Medical Diagnosis</span>
        </p>
        <p style='margin: 10px 0 0 0; color: #7f8c8d; font-size: 0.8em;'>
            © 2024-2025 All Rights Reserved by Jerin_Papri_Mithila
        </p>
    </div>
""", unsafe_allow_html=True)