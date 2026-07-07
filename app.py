"""
Advanced Titanic Survival Analytics & Prediction Engine
Production Quality Architecture - Monolithic Streamlit Cloud Compliant File
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as io
from plotly.subplots import make_subplots
import os
import joblib

# ML Diagnostics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ------------------------------------------------------------------
# CONFIGURATION & STATIC ASSETS
# ------------------------------------------------------------------
st.set_page_config(page_title="Titanic Predictive Operations", layout="wide", initial_sidebar_state="expanded")

# Inject Custom Styling Directly
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0b0f19 0%, #111827 100%); color: #f3f4f6; }
    div[data-testid="stMetricBlock"] {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 14px !important;
        padding: 15px !important;
    }
    .css-1dan26b { background-color: rgba(255,255,255,0.05); }
</style>
""", unsafe_allow_html=True)

# Helper function to cache structured data load
@st.cache_data
def load_and_clean_data(file_path="titanic.csv"):
    if not os.path.exists(file_path):
        # Fallback dummy generator if dataset missing
        np.random.seed(42)
        records = 891
        return pd.DataFrame({
            'PassengerId': range(1, records+1),
            'Survived': np.random.choice([0, 1], size=records, p=[0.616, 0.384]),
            'Pclass': np.random.choice([1, 2, 3], size=records, p=[0.24, 0.21, 0.55]),
            'Name': [f"Passenger, Passenger {i}" for i in range(1, records+1)],
            'Sex': np.random.choice(['male', 'female'], size=records, p=[0.64, 0.36]),
            'Age': np.random.normal(29.7, 14.5, size=records).clip(0.4, 80.0),
            'SibSp': np.random.choice([0, 1, 2, 3, 4, 5], size=records, p=[0.68, 0.23, 0.04, 0.02, 0.02, 0.01]),
            'Parch': np.random.choice([0, 1, 2, 3], size=records, p=[0.76, 0.13, 0.09, 0.02]),
            'Ticket': [f"PC {np.random.randint(10000, 99999)}" for _ in range(records)],
            'Fare': np.random.exponential(32.2, size=records).clip(0, 512.3),
            'Cabin': np.random.choice([np.nan, 'C85', 'E46', 'G6'], size=records, p=[0.77, 0.10, 0.08, 0.05]),
            'Embarked': np.random.choice(['S', 'C', 'Q'], size=records, p=[0.72, 0.20, 0.08])
        })
    df = pd.read_csv(file_path)
    # Inline Feature Generation
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = np.where(df['FamilySize'] == 1, 1, 0)
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss').replace('Mme', 'Mrs').fillna('Unknown')
    return df

df_raw = load_and_clean_data()

# ------------------------------------------------------------------
# NAVIGATION SIDEBAR LAYER
# ------------------------------------------------------------------
with st.sidebar:
    st.title("🚢 Titanic ML Ops")
    st.markdown("---")
    page = st.radio("Enterprise Navigation Center", [
        "🏠 Home Base", 
        "📊 Analytical Dashboard", 
        "🔍 Data Explorer Core", 
        "🧪 Machine Learning Engine", 
        "🔮 Real-time Prediction", 
        "💡 Deep Domain Insights"
    ])
    st.markdown("---")
    st.caption("Environment Focus: Streamlit Community Production Cloud Layer v2.1")

# ------------------------------------------------------------------
# 1. HOME PAGE
# ------------------------------------------------------------------
if page == "🏠 Home Base":
    st.title("Enterprise Titanic Survival Predictive Optimization Engine")
    st.subheader("Production grade processing, pipeline diagnostics, and validation dashboard.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Strategic Project Framework
        This operations terminal unifies advanced feature engineering frameworks and state-of-the-art classifier suites to resolve binary evaluation constraints on historical manifests.
        
        #### Technical Capabilities Included
        * **Dynamic Column Preprocessors**: Automated handling of missing categorical maps & continuous vector transformations.
        * **Hyperparameter Benchmarking**: Real-time evaluation of 7 industry-standard classification models.
        * **Operational Explainability**: Feature importance profiling, confusion arrays, and interactive prediction engines.
        """)
    with col2:
        st.info("💡 **Operational Advisory:** Use the sidebar to step sequentially through exploration, automated machine learning pipelines, and production deployments.")

# ------------------------------------------------------------------
# 2. ANALYTICAL DASHBOARD
# ------------------------------------------------------------------
elif page == "📊 Analytical Dashboard":
    st.title("Operational Intelligence Center")
    
    # KPIs calculations
    tot_passengers = int(df_raw.shape[0])
    survived_cnt = int(df_raw['Survived'].sum())
    death_cnt = tot_passengers - survived_cnt
    surv_rate = (survived_cnt / tot_passengers) * 100
    avg_age = float(df_raw['Age'].mean())
    avg_fare = float(df_raw['Fare'].mean())
    
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    kpi1.metric("Total Passengers", f"{tot_passengers:,}")
    kpi2.metric("Survivors", f"{survived_cnt:,}", delta=f"+{surv_rate:.1f}%")
    kpi3.metric("Deaths", f"{death_cnt:,}", delta=f"-{100-surv_rate:.1f}%", delta_color="inverse")
    kpi4.metric("Survival Rate", f"{surv_rate:.2f}%")
    kpi5.metric("Avg Passenger Age", f"{avg_age:.1f} Yrs")
    kpi6.metric("Avg Ticket Fare", f"${avg_fare:.2f}")
    
    st.markdown("---")
    
    # Layout Grid for Charts
    c1, c2 = st.columns(2)
    with c1:
        # Pclass distribution
        fig1 = px.histogram(df_raw, x="Pclass", color="Survived", barmode="group", title="Survival Split across Cabin Classes", template="plotly_dark")
        st.plotly_chart(fig1, use_container_width=True)
        
        # Age distribution
        fig2 = px.histogram(df_raw, x="Age", color="Survived", marginal="rug", title="Passenger Cohort Age Profiling", template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)
        
        # Pie Distribution
        fig3 = px.pie(df_raw, names="Embarked", title="Embarkation Port Segment Breakdown", template="plotly_dark", hole=0.4)
        st.plotly_chart(fig3, use_container_width=True)

        # Violin
        fig_viol = px.violin(df_raw, y="Age", x="Pclass", color="Survived", box=True, points="all", title="Age Violin vs Ticket Class Distribution", template="plotly_dark")
        st.plotly_chart(fig_viol, use_container_width=True)

    with c2:
        # Gender Split
        fig4 = px.histogram(df_raw, x="Sex", color="Survived", barmode="group", title="Gender Biased Survival Disparity Matrix", template="plotly_dark")
        st.plotly_chart(fig4, use_container_width=True)
        
        # Fare distribution
        fig5 = px.histogram(df_raw, x="Fare", color="Survived", log_y=True, title="Fare Distribution Spectrum (Log-Scaled Axis)", template="plotly_dark")
        st.plotly_chart(fig5, use_container_width=True)
        
        # Treemap
        fig6 = px.treemap(df_raw, path=['Pclass', 'Sex', 'Embarked'], title="Hierarchical Distribution Matrix Breakdown", template="plotly_dark")
        st.plotly_chart(fig6, use_container_width=True)

        # Box plot
        fig_box = px.box(df_raw, x="Survived", y="Fare", log_y=True, title="Fare Variance Logarithmic Box Array", template="plotly_dark")
        st.plotly_chart(fig_box, use_container_width=True)

# ------------------------------------------------------------------
# 3. DATA EXPLORER CORE
# ------------------------------------------------------------------
elif page == "🔍 Data Explorer Core":
    st.title("Data Asset Explorer Hub")
    
    # Filter operations
    st.sidebar.markdown("### Interactive DataFrame Tuning")
    sex_sel = st.sidebar.multiselect("Filter Sex Grouping", df_raw['Sex'].unique(), default=df_raw['Sex'].unique())
    class_sel = st.sidebar.multiselect("Filter Cabin Tier", df_raw['Pclass'].unique(), default=df_raw['Pclass'].unique())
    
    filtered_df = df_raw[(df_raw['Sex'].isin(sex_sel)) & (df_raw['Pclass'].isin(class_sel))]
    
    tabs1, tabs2 = st.tabs(["🗃 Filtered Ledger Matrix", "📈 Structural Statistics"])
    with tabs1:
        st.dataframe(filtered_df, use_container_width=True)
        csv_buffer = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Filtered Slice as CSV", data=csv_buffer, file_name="filtered_titanic.csv", mime="text/csv")
        
    with tabs2:
        st.markdown("#### High-Level Descriptive Variables")
        st.dataframe(filtered_df.describe(include='all').fillna('-'), use_container_width=True)
        
        st.markdown("#### Missing Information Architecture Mapping")
        null_counts = filtered_df.isnull().sum().to_frame(name="Missing Elements Count")
        null_counts['% of Total'] = (null_counts['Missing Elements Count'] / len(filtered_df)) * 100
        st.table(null_counts)

# ------------------------------------------------------------------
# 4. MACHINE LEARNING ENGINE
# ------------------------------------------------------------------
elif page == "🧪 Machine Learning Engine":
    st.title("Algorithmic Engine & Benchmark Suite")
    
    # Build clean model ready matrices
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'SibSp', 'Parch', 'FamilySize', 'Title']
    X = df_raw[features]
    y = df_raw['Survived']
    
    numeric_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
    numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_features = ['Sex', 'Embarked', 'Pclass', 'Title']
    categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    
    preprocessor = ColumnTransformer([('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(max_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(probability=True),
        'Naive Bayes': GaussianNB()
    }
    
    results = []
    trained_pipelines = {}
    
    for name, model in models.items():
        pipe = Pipeline([('prep', preprocessor), ('clf', model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds
        
        results.append({
            'Algorithm Suite': name,
            'Accuracy': accuracy_score(y_test, preds),
            'Precision': precision_score(y_test, preds, zero_division=0),
            'Recall': recall_score(y_test, preds, zero_division=0),
            'F1 Score': f1_score(y_test, preds, zero_division=0),
            'ROC AUC': roc_auc_score(y_test, probs)
        })
        trained_pipelines[name] = pipe

    res_df = pd.DataFrame(results)
    st.markdown("### Algorithmic Evaluation Comparison Benchmark")
    st.dataframe(res_df.style.highlight_max(axis=0, subset=['Accuracy', 'F1 Score', 'ROC AUC'], color='#1e3a8a'), use_container_width=True)
    
    # Choose Model for deep metrics focus
    selected_eval_model = st.selectbox("Choose Trained Architecture for Inspection Diagram", res_df['Algorithm Suite'].tolist())
    
    # Extract targeted diagnostics
    tgt_pipe = trained_pipelines[selected_eval_model]
    tgt_preds = tgt_pipe.predict(X_test)
    tgt_probs = tgt_pipe.predict_proba(X_test)[:, 1] if hasattr(tgt_pipe.named_steps['clf'], "predict_proba") else tgt_preds
    
    cm = confusion_matrix(y_test, tgt_preds)
    
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown(f"#### {selected_eval_model} Confusion Matrix")
        fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted Class Target", y="Actual Verified Class"), x=['Perished', 'Survived'], y=['Perished', 'Survived'], color_continuous_scale="Viridis")
        st.plotly_chart(fig_cm, use_container_width=True)
    with mc2:
        st.markdown("#### ROC Binary Diagnostic Plot")
        fpr, tpr, _ = roc_curve(y_test, tgt_probs)
        fig_roc = px.area(x=fpr, y=tpr, title=f"ROC Curve (AUC: {roc_auc_score(y_test, tgt_probs):.3f})", labels=dict(x="False Positive Rate", y="True Positive Rate"))
        fig_roc.add_shape(type="line", line=dict(dash='dash', color="red"), x0=0, x1=1, y0=0, y1=1)
        st.plotly_chart(fig_roc, use_container_width=True)

# ------------------------------------------------------------------
# 5. REAL-TIME PREDICTION
# ------------------------------------------------------------------
elif page == "🔮 Real-time Prediction":
    st.title("Inference Engine & Probability Scoring Matrix")
    
    # Direct pipeline generation for execution accuracy assurance
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'SibSp', 'Parch', 'FamilySize', 'Title']
    X = df_raw[features]
    y = df_raw['Survived']
    
    numeric_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
    numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_features = ['Sex', 'Embarked', 'Pclass', 'Title']
    categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer([('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])
    
    # Enforce industrial best architecture
    active_pipeline = Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))])
    active_pipeline.fit(X, y)
    
    with st.form("inference_form"):
        st.markdown("### Profile Feature Vector Parameter Matrix Input")
        p1, p2, p3 = st.columns(3)
        with p1:
            p_class = st.selectbox("Ticket Tier Class (Pclass)", [1, 2, 3], index=2)
            sex = st.selectbox("Gender Group Allocation", ["male", "female"], index=0)
            emb = st.selectbox("Port Coordinates Matrix", ["S", "C", "Q"], index=0)
        with p2:
            age = st.slider("Age Variable Index", 0.5, 90.0, 30.0, step=0.5)
            fare = st.number_input("Transaction Fare Value ($)", min_value=0.0, max_value=512.5, value=15.0)
        with p3:
            sib = st.number_input("SibSp (Siblings/Spouse Count)", min_value=0, max_value=10, value=0)
            parch = st.number_input("Parch (Parents/Children Count)", min_value=0, max_value=10, value=0)
            
        submit_inf = st.form_submit_button("⚡ Execute Matrix Prediction Routine")
        
        if submit_inf:
            f_size = sib + parch + 1
            title_calc = "Mr" if sex == "male" else "Mrs"
            
            # Form single-line payload vector
            payload = pd.DataFrame([{
                'Pclass': p_class, 'Sex': sex, 'Age': age, 'Fare': fare,
                'Embarked': emb, 'SibSp': sib, 'Parch': parch, 'FamilySize': f_size, 'Title': title_calc
            }])
            
            prediction = active_pipeline.predict(payload)[0]
            probability = active_pipeline.predict_proba(payload)[0][1]
            
            st.markdown("---")
            if prediction == 1:
                st.success(f"🏆 Inference Resolved: **Passenger Survived** with a raw certainty score of **{probability*100:.1f}%**")
                st.balloons()
            else:
                st.error(f"💀 Inference Resolved: **Passenger Perished** with a raw certainty score of **{(1-probability)*100:.1f}%**")

# ------------------------------------------------------------------
# 6. DEEP DOMAIN INSIGHTS
# ------------------------------------------------------------------
elif page == "💡 Deep Domain Insights":
    st.title("Strategic Business Intelligence & Analytical Audits")
    
    # Form processing blocks
    f_df = df_raw[df_raw['Sex'] == 'female']
    m_df = df_raw[df_raw['Sex'] == 'male']
    child_df = df_raw[df_raw['Age'] < 16]
    rich_df = df_raw[df_raw['Fare'] >= df_raw['Fare'].quantile(0.75)]
    
    ic1, ic2, ic3, ic4 = st.columns(4)
    ic1.metric("Female Survival", f"{(f_df['Survived'].sum()/len(f_df))*100:.1f}%")
    ic2.metric("Male Survival", f"{(m_df['Survived'].sum()/len(m_df))*100:.1f}%")
    ic3.metric("Child Cohort Survival", f"{(child_df['Survived'].sum()/len(child_df))*100:.1f}%")
    ic4.metric("Affluent Bracket Survival", f"{(rich_df['Survived'].sum()/len(rich_df))*100:.1f}%")
    
    st.markdown("### Primary Domain Observations")
    st.markdown("""
    * **The Gender Safety Imperative**: Female allocation manifests a massive statistical bias advantage, out-surviving male attributes significantly due to historical maritime structural protocols ("Women and children first").
    * **Class & Capital Stratification**: Passages listed in Class 1 achieved premium survivability rates relative to baseline Class 3 passengers, asserting socioeconomic privilege correlates to structural hazard safety placement.
    """)
