import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="wide")

@st.cache_resource
def load_model():
    try:
        with open('churn_prediction_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('feature_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        features = pd.read_csv('model_features.csv')['feature'].tolist()
        return model, scaler, features
    except Exception as e:
        return None, None, None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('processed_customer_data.csv')
        if df['Churn'].dtype == 'object':
            df['Churn'] = (df['Churn'] == 'Yes').astype(int)
        return df
    except:
        return None

st.title("📊 Customer Churn Prediction System")
st.markdown("### Real-time Churn Risk Assessment")
st.markdown("---")

st.sidebar.header("Navigation")
app_mode = st.sidebar.selectbox("Select Mode", 
    ["Dashboard Overview", "Single Prediction", "Model Insights"])

if app_mode == "Dashboard Overview":
    df = load_data()
    
    if df is not None:
        st.header("Business Intelligence Dashboard")
        
        total = len(df)
        churned = int(df['Churn'].sum())
        churn_rate = (churned / total) * 100
        avg_revenue = float(df['MonthlyCharges'].mean())
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", f"{total:,}")
        col2.metric("Churned", f"{churned:,}")
        col3.metric("Churn Rate", f"{churn_rate:.1f}%")
        col4.metric("Avg Revenue", f"${avg_revenue:.2f}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Revenue Distribution")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df['MonthlyCharges'], nbinsx=30, marker_color='#3498db'))
            fig.update_layout(xaxis_title="Monthly Charges ($)", yaxis_title="Customers", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Tenure Distribution")
            churned_df = df[df['Churn'] == 1]
            retained_df = df[df['Churn'] == 0]
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=retained_df['tenure'], name='Retained', 
                                      marker_color='#2ecc71', opacity=0.7, nbinsx=30))
            fig.add_trace(go.Histogram(x=churned_df['tenure'], name='Churned', 
                                      marker_color='#e74c3c', opacity=0.7, nbinsx=30))
            fig.update_layout(barmode='overlay', xaxis_title="Tenure (months)", 
                            yaxis_title="Customers", height=400)
            st.plotly_chart(fig, use_container_width=True)

elif app_mode == "Single Prediction":
    st.header("Customer Churn Risk Assessment")
    
    model, scaler, features = load_model()
    
    if model:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
        
        with col2:
            st.subheader("Services")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            internet = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
            phone = st.selectbox("Phone", ["No", "Yes"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        
        with col3:
            st.subheader("Billing")
            monthly = st.number_input("Monthly Charges ($)", 0.0, 150.0, 50.0)
            total = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0)
            payment = st.selectbox("Payment Method", 
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", 
                 "Credit card (automatic)"])
            paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
        
        if st.button("Predict Churn Risk", type="primary"):
            input_data = {
                'gender': gender, 'SeniorCitizen': 1 if senior == "Yes" else 0,
                'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
                'PhoneService': phone, 'InternetService': internet,
                'Contract': contract, 'PaperlessBilling': paperless,
                'PaymentMethod': payment, 'MonthlyCharges': monthly,
                'TotalCharges': total, 'CustomerValue': total / (tenure + 1),
                'ActiveServices': 2, 
                'FamilyAccount': 1 if partner == "Yes" or dependents == "Yes" else 0,
                'HighRiskSenior': 1 if senior == "Yes" and partner == "No" and dependents == "No" else 0,
                'RevenueConcentration': monthly / (total + 1)
            }
            
            input_df = pd.DataFrame([input_data])
            input_encoded = pd.get_dummies(input_df, drop_first=True)
            
            for col in features:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            
            input_encoded = input_encoded[features]
            
            prediction = model.predict(input_encoded)[0]
            probability = model.predict_proba(input_encoded)[0]
            
            st.markdown("---")
            st.subheader("Prediction Results")
            
            churn_prob = probability[1] * 100
            risk_level = "HIGH" if churn_prob > 70 else "MEDIUM" if churn_prob > 40 else "LOW"
            risk_color = "#e74c3c" if risk_level == "HIGH" else "#f39c12" if risk_level == "MEDIUM" else "#2ecc71"
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"### Churn Probability")
                st.markdown(f"# {churn_prob:.1f}%")
                st.markdown(f"<h2 style='color: {risk_color};'>Risk: {risk_level}</h2>", 
                           unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"### Predicted Status")
                status = "⚠️ CHURN RISK" if prediction == 1 else "✅ LIKELY TO STAY"
                st.markdown(f"## {status}")
                st.markdown(f"### Customer Value")
                st.markdown(f"## ${total:.2f}")
            
            with col3:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=churn_prob,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={'axis': {'range': [None, 100]}, 'bar': {'color': risk_color},
                           'steps': [{'range': [0, 40], 'color': "#d5f4e6"},
                                    {'range': [40, 70], 'color': "#fef5e7"},
                                    {'range': [70, 100], 'color': "#fadbd8"}]}
                ))
                fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Recommended Actions")
            
            if churn_prob > 70:
                st.error("⚠️ HIGH RISK: Immediate action required")
                st.markdown("""
                1. **Urgent**: Personal call from account manager within 24 hours
                2. **Offer**: Contract upgrade with 20% discount
                3. **Review**: Address any outstanding service quality issues
                4. **Incentive**: Provide loyalty reward or promotional offer
                """)
            elif churn_prob > 40:
                st.warning("⚠️ MODERATE RISK: Proactive engagement needed")
                st.markdown("""
                1. **Schedule**: Customer check-in call within 2 weeks
                2. **Evaluate**: Review service usage and recommend optimizations
                3. **Offer**: Contract extension with added benefits
                4. **Educate**: Ensure customer awareness of all available services
                """)
            else:
                st.success("✅ LOW RISK: Customer appears stable")
                st.markdown("""
                1. **Maintain**: Continue standard engagement protocols
                2. **Upsell**: Consider offering additional service packages
                3. **Monitor**: Maintain high service quality standards
                4. **Survey**: Include in next quarterly satisfaction survey
                """)

elif app_mode == "Model Insights":
    st.header("Model Performance & Feature Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Comparison")
        st.markdown("""
        | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
        |-------|----------|-----------|--------|----------|---------|
        | **Logistic Regression** | **80.8%** | **67.6%** | **53.5%** | **59.7%** | **84.7%** |
        | Random Forest | 80.3% | 66.7% | 51.9% | 58.4% | 84.2% |
        | Gradient Boosting | 80.2% | 66.1% | 52.1% | 58.3% | 84.2% |
        """)
        st.info("**Best Model:** Logistic Regression with 84.7% ROC-AUC")
    
    with col2:
        st.subheader("Top Predictive Features")
        features_df = pd.DataFrame({
            'Feature': ['Revenue Concentration', 'Tenure', 'Total Charges', 
                       'Monthly Charges', 'Customer Value', 'Fiber Optic',
                       'Two Year Contract', 'Electronic Check'],
            'Importance': [18.4, 11.9, 8.3, 8.2, 7.6, 7.3, 5.4, 4.2]
        })
        
        fig = go.Figure(go.Bar(
            x=features_df['Importance'],
            y=features_df['Feature'],
            orientation='h',
            marker_color='#3498db',
            text=features_df['Importance'].apply(lambda x: f'{x}%'),
            textposition='outside'
        ))
        fig.update_layout(
            xaxis_title="Importance (%)",
            yaxis_title="",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Business Impact Assessment")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Churn Detection Rate", "51.9%", "194 of 374 churners identified")
    col2.metric("Model Accuracy", "80.8%", "1,139 of 1,409 correct predictions")
    col3.metric("Potential Annual Savings", "$45,230", "Based on 30% retention success")
    col4.metric("Avg Customer Value", "$777", "Annual revenue per customer")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #7f8c8d;'><p>Customer Churn Prediction System | Machine Learning Analytics</p></div>", 
            unsafe_allow_html=True)
