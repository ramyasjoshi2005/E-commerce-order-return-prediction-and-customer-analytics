import streamlit as st
import pandas as pd
import sqlite3
import joblib
import json
import shap
import matplotlib.pyplot as plt
import os
import numpy as np

# Set page config
st.set_page_config(page_title="Product Return Analytics", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')

@st.cache_resource
def load_resources():
    features_path = os.path.join(BASE_DIR, 'data', 'processed', 'features.csv')
    model_path = os.path.join(BASE_DIR, 'models', 'best_model.joblib')
    
    features = pd.read_csv(features_path)
    model = joblib.load(model_path)
    
    with open(os.path.join(ARTIFACTS_DIR, 'test_metrics.json'), 'r') as f:
        metrics = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'class_distribution.json'), 'r') as f:
        class_dist = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'curve_data.json'), 'r') as f:
        curve_data = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'dataset_stats.json'), 'r') as f:
        stats = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'business_insight.json'), 'r') as f:
        insight = json.load(f)
        
    val_df = pd.read_csv(os.path.join(ARTIFACTS_DIR, 'validation_comparison.csv'))
    audit = pd.read_csv(os.path.join(ARTIFACTS_DIR, 'leakage_audit.csv'))
    tuning_df = pd.read_csv(os.path.join(ARTIFACTS_DIR, 'threshold_tuning.csv'))
    
    return features, model, metrics, class_dist, val_df, audit, curve_data, stats, insight, tuning_df

try:
    features, model, metrics, class_dist, val_df, audit, curve_data, dataset_stats, business_insight, tuning_df = load_resources()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview", 
    "Exploratory Analytics",
    "Return Prediction", 
    "Prediction Explanation",
    "Model Evaluation", 
    "Interview Notes"
])

with tab1:
    st.title("Product Return Prediction & Root-Cause Analytics")
    st.markdown("#### *Predicting return risk at order placement using leakage-safe temporal modeling.*")
    
    st.markdown("---")
    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
    mcol1.metric("Total Orders", f"{dataset_stats['total_rows']:,}")
    mcol2.metric("Return Prevalence", f"{dataset_stats['overall_return_rate']*100:.1f}%")
    mcol3.metric("Test ROC-AUC", f"{metrics['roc_auc']:.3f}")
    mcol4.metric("Test PR-AUC", f"{metrics['pr_auc']:.3f}")
    mcol5.metric("Decision Threshold", f"{metrics['threshold']:.2f}")

    st.markdown("---")
    st.subheader("💡 KEY BUSINESS INSIGHT")
    
    hr_rate = business_insight['high_risk_group']['rate'] * 100
    lr_rate = business_insight['low_risk_group']['rate'] * 100
    risk_ratio = business_insight['risk_ratio']
    
    st.success(f"""
    **OBSERVED:** Customers with a historical return rate >15% were observed to have a **{hr_rate:.1f}%** return rate on subsequent orders, compared to only **{lr_rate:.1f}%** for those ≤15% (a **{risk_ratio:.1f}×** difference).
    
    **MODEL:** The model assigns substantial predictive importance to this historical behavioral pattern.
    
    **BUSINESS ACTION:** Orders belonging to this high-risk segment are a potential intervention point (e.g., a 12-hour hold queue for manual review before shipping costs are incurred). This is a hypothetical intervention worth testing via randomized experiment.
    """)
    
    st.markdown("---")
    colA, colB = st.columns(2)
    with colA:
        st.subheader("WHY THIS MODEL?")
        st.markdown("""
        **XGBoost** was selected because it natively handles the interactions between categorical dimensions and historical aggregates. 
        It achieved the highest Validation PR-AUC and strong Brier Score calibration. `scale_pos_weight` natively handles class imbalance without using synthetic oversampling (SMOTE), which avoids temporal leakage.
        """)
        
    with colB:
        st.subheader("DATA / EVALUATION DESIGN")
        st.markdown(f"""
        Strict chronological splitting simulates a real-world production deployment:
        * **Train:** 2018–2019
        * **Validation:** 2020 
        * **Test:** 2021
        
        **🚨 No post-order information used.** All features are strictly calculated using information available at order placement.
        """)

with tab2:
    st.header("Exploratory Analytics")
    st.write("Distribution of returns by year (showing stability of the target).")
    st.bar_chart(dataset_stats['returns_by_year'])
    st.write(f"**Date Range:** {dataset_stats['min_date']} to {dataset_stats['max_date']}")
    
    st.subheader("Class Distribution (Chronological Split)")
    dist_df = pd.DataFrame(class_dist).T
    dist_df.columns = ['0 (Keep)', '1 (Return)']
    dist_df['Return %'] = dist_df['1 (Return)'] / (dist_df['0 (Keep)'] + dist_df['1 (Return)']) * 100
    st.table(dist_df.style.format({'Return %': '{:.2f}%'}))
    
    st.subheader("Target Leakage Audit")
    st.table(audit)

with tab3:
    st.header("Predict Return Risk")
    st.write("Simulate an order placement. The system will dynamically calculate the customer's history up to the simulated order date to prevent leakage.")
    
    conn = sqlite3.connect(os.path.join(BASE_DIR, 'data', 'processed', 'superstore.db'))
    customers = pd.read_sql("SELECT DISTINCT customer_id FROM orders LIMIT 100", conn)['customer_id'].tolist()
    
    col_order, col_hist = st.columns(2)
    with col_order:
        st.subheader("Order Information")
        customer_id = st.selectbox("Select Customer (Simulation)", customers)
        order_date = st.date_input("Simulated Order Date", pd.to_datetime('2021-08-17').date())
        
        market = st.selectbox("Market", features['market'].unique())
        region = st.selectbox("Region", features['region'].unique())
        country = st.selectbox("Country", features['country'].unique())
        segment = st.selectbox("Segment", features['segment'].unique())
        order_priority = st.selectbox("Order Priority", features['order_priority'].unique())
        total_sales = st.number_input("Total Sales ($)", 0.0, 10000.0, 100.0)
        total_quantity = st.number_input("Total Quantity", 1, 100, 2)
        avg_discount = st.number_input("Average Discount", 0.0, 1.0, 0.0)
        max_discount = st.number_input("Max Discount", 0.0, 1.0, 0.0)
        number_of_products = st.number_input("Number of Products", 1, 50, 1)
        number_of_categories = st.number_input("Number of Categories", 1, 10, 1)
        
    with col_hist:
        st.subheader("Dynamically Calculated Point-in-Time History")
        
        # Calculate dynamic history
        hist_query = f"""
        SELECT 
            COUNT(o.order_id) as hist_orders,
            SUM(CASE WHEN r.order_id IS NOT NULL THEN 1 ELSE 0 END) as hist_returns,
            MAX(o.order_date) as last_order_date
        FROM orders o
        LEFT JOIN returns r ON o.order_id = r.order_id
        WHERE o.customer_id = '{customer_id}' AND o.order_date < '{order_date}'
        """
        hist_df = pd.read_sql(hist_query, conn)
        
        historical_orders = hist_df['hist_orders'].iloc[0]
        hist_returns = hist_df['hist_returns'].iloc[0]
        last_order = hist_df['last_order_date'].iloc[0]
        
        if pd.isna(last_order):
            days_since_prev = 9999
        else:
            days_since_prev = (pd.to_datetime(order_date) - pd.to_datetime(last_order)).days
            
        # Laplace smoothing parameters used in training
        prior_returns = 1.0
        prior_orders = 7.0
        cust_hist_rate = (hist_returns + prior_returns) / (historical_orders + prior_orders)
        
        st.metric("Historical Orders Before Date", historical_orders)
        st.metric("Days Since Previous Order", days_since_prev if days_since_prev != 9999 else "First Order")
        st.metric("Customer Historical Return Rate (Smoothed)", f"{cust_hist_rate:.3f}")
        
        order_month = pd.to_datetime(order_date).month
        is_weekend = 1 if pd.to_datetime(order_date).dayofweek in [5, 6] else 0

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("PREDICT RETURN RISK", type="primary"):
        input_data = pd.DataFrame([{
            'market': market,
            'region': region,
            'country': country,
            'segment': segment,
            'order_priority': order_priority,
            'total_sales': total_sales,
            'total_quantity': total_quantity,
            'average_discount': avg_discount,
            'max_discount': max_discount,
            'number_of_products': number_of_products,
            'number_of_categories': number_of_categories,
            'historical_orders': historical_orders,
            'days_since_previous_order': days_since_prev,
            'customer_historical_return_rate': cust_hist_rate,
            'order_month': order_month,
            'order_day_of_week': pd.to_datetime(order_date).dayofweek,
            'is_weekend': is_weekend
        }])
        
        prob = model.predict_proba(input_data)[0][1]
        threshold = metrics['threshold']
        
        st.markdown("---")
        st.subheader("Prediction Result")
        
        rcol1, rcol2, rcol3 = st.columns(3)
        rcol1.metric("Return Probability", f"{prob * 100:.1f}%")
        rcol2.metric("Risk Level", "HIGH" if prob >= threshold else "LOW", delta="-Flagged" if prob >= threshold else "+Clear", delta_color="inverse")
        rcol3.metric("Decision Threshold", f"{threshold * 100:.0f}%")
        
        st.session_state['last_input'] = input_data
        st.session_state['last_prob'] = prob
        
    conn.close()

with tab4:
    st.header("Prediction Explanation")
    st.write("This section shows the statistical associations driving the model's prediction. It does **not** prove causality.")
    
    if 'last_input' in st.session_state:
        st.write("### Factors influencing the current prediction")
        
        preprocessor = model.named_steps['preprocessor']
        classifier = model.named_steps['classifier']
        input_processed = preprocessor.transform(st.session_state['last_input'])
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(input_processed)
        
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
        num_features = preprocessor.transformers_[0][2]
        feature_names = list(num_features) + list(cat_features)
        
        shap_df = pd.DataFrame({'Feature': feature_names, 'SHAP Value': shap_values[0]})
        shap_df['Absolute'] = shap_df['SHAP Value'].abs()
        shap_df = shap_df.sort_values('Absolute', ascending=False).head(5)
        
        for idx, row in shap_df.iterrows():
            direction = "Increased Risk" if row['SHAP Value'] > 0 else "Decreased Risk"
            color = "red" if row['SHAP Value'] > 0 else "green"
            st.markdown(f"- **{row['Feature']}** (:<span style='color:{color}'>{direction}</span>>)", unsafe_allow_html=True)
            
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.barplot(shap_values[0], max_display=10, feature_names=feature_names, show=False)
        st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("Global Model Feature Importance")
    if st.button("Generate Global Explanations"):
        with st.spinner("Calculating SHAP..."):
            preprocessor = model.named_steps['preprocessor']
            classifier = model.named_steps['classifier']
            sample_X = features.drop(columns=['order_id', 'customer_id', 'order_date', 'returned', 'historical_returns', 'market_historical_returns'], errors='ignore').sample(500, random_state=42)
            sample_processed = preprocessor.transform(sample_X)
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(sample_processed)
            cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
            num_features = preprocessor.transformers_[0][2]
            feature_names = list(num_features) + list(cat_features)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, sample_processed, feature_names=feature_names, show=False)
            st.pyplot(fig)

with tab5:
    st.header("Model Evaluation")
    
    st.subheader("Validation Set Performance (Model Selection)")
    st.dataframe(val_df.style.highlight_max(subset=['pr_auc', 'f1', 'roc_auc'], color='lightgreen').highlight_min(subset=['brier_score'], color='lightgreen'))
    
    st.subheader("Threshold Optimization (Validation Set)")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(tuning_df['threshold'], tuning_df['f1'], label='F1')
    ax.plot(tuning_df['threshold'], tuning_df['precision'], label='Precision')
    ax.plot(tuning_df['threshold'], tuning_df['recall'], label='Recall')
    ax.axvline(metrics['threshold'], color='k', linestyle='--', label=f"Selected Threshold ({metrics['threshold']:.2f})")
    ax.legend()
    st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("Final Test Performance (Untouched 2021 Data)")
    test_metrics_df = pd.DataFrame([metrics])
    st.table(test_metrics_df[['model', 'roc_auc', 'pr_auc', 'f1', 'precision', 'recall', 'brier_score', 'threshold']].style.format(precision=3))
    
    st.markdown("### Evaluation Curves (Test Set)")
    col1, col2, col3 = st.columns(3)
    with col1:
        fig, ax = plt.subplots()
        ax.plot(curve_data['test']['roc']['fpr'], curve_data['test']['roc']['tpr'])
        ax.plot([0,1], [0,1], 'k--')
        ax.set_title("ROC Curve")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        ax.plot(curve_data['test']['pr']['recall'], curve_data['test']['pr']['precision'])
        ax.set_title("Precision-Recall Curve")
        st.pyplot(fig)
    with col3:
        fig, ax = plt.subplots()
        ax.plot(curve_data['test']['calibration']['prob_pred'], curve_data['test']['calibration']['prob_true'], marker='o')
        ax.plot([0,1], [0,1], 'k--')
        ax.set_title("Calibration Curve")
        st.pyplot(fig)

with tab6:
    st.header("Interview Notes")
    st.markdown("""
    **1. Why this prediction point?**  
    We predict exactly at "Order Placement" because it is the most actionable moment. If we wait until shipping data is available, shipping costs are already incurred.

    **2. How did you prevent target leakage?**  
    All historical features were calculated using `cumsum()` shifted by 1 chronologically. Post-order information (profit, shipping cost) was explicitly dropped. 

    **3. Why temporal splitting?**  
    E-commerce behavior changes over time. Randomly splitting data (K-Fold) would leak future trends into the training set, yielding artificially high performance.

    **4. Why this model?**  
    XGBoost naturally handles the non-linear interactions between categorical features and historical aggregates without complex feature engineering, achieving the best PR-AUC.

    **5. Why PR-AUC instead of accuracy?**  
    Our dataset is imbalanced (~14.5% returns). Accuracy is misleading because predicting "Not Returned" for everything yields 85.5% accuracy but zero business value.

    **6. How did you handle class imbalance?**  
    We used `scale_pos_weight` in XGBoost rather than SMOTE. SMOTE synthesizes data between nearest neighbors, which is dangerous in time-series data because it can blend future and past behaviors.

    **7. How was the threshold selected?**  
    The decision threshold was optimized by sweeping from 0.05 to 0.95 on the **Validation Set (2020)** to maximize F1-score. We then locked it and applied it exactly once to the Test Set.

    **8. Why SHAP?**  
    SHAP provides consistent, model-agnostic feature attributions. It allows us to explain the model's math, building trust with human operators.

    **9. What does SHAP NOT prove?**  
    SHAP proves **association**, not **causation**. The model relying heavily on "Discount" mathematically does not prove the discount *caused* the return.

    **10. What is the business action enabled by the model?**  
    High-risk orders can be routed to a manual review queue, held for 12 hours to allow customer cancellation, or flagged for proactive customer service outreach.
    """)
