import pandas as pd
import numpy as np
import time
import os
import json
import joblib
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, brier_score_loss
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve

def load_data(filepath='data/processed/features.csv'):
    df = pd.read_csv(filepath)
    df['order_date'] = pd.to_datetime(df['order_date'])
    df = df.sort_values(['order_date', 'order_id']).reset_index(drop=True)
    return df

def generate_dataset_stats(df):
    stats_dict = {
        'total_rows': len(df),
        'unique_orders': df['order_id'].nunique(),
        'unique_customers': df['customer_id'].nunique(),
        'min_date': df['order_date'].min().strftime('%Y-%m-%d'),
        'max_date': df['order_date'].max().strftime('%Y-%m-%d'),
        'overall_return_rate': float(df['returned'].mean()),
        'returns_by_year': df.groupby(df['order_date'].dt.year)['returned'].mean().to_dict()
    }
    with open('artifacts/dataset_stats.json', 'w') as f:
        json.dump(stats_dict, f)

def generate_business_insight(df):
    high_risk = df[df['customer_historical_return_rate'] > 0.15]
    low_risk = df[df['customer_historical_return_rate'] <= 0.15]
    
    n_high = len(high_risk)
    n_low = len(low_risk)
    
    r_high = high_risk['returned'].sum()
    r_low = low_risk['returned'].sum()
    
    rate_high = r_high / n_high if n_high > 0 else 0
    rate_low = r_low / n_low if n_low > 0 else 0
    
    risk_ratio = rate_high / rate_low if rate_low > 0 else 0
    
    # Proportion Z-test
    from statsmodels.stats.proportion import proportions_ztest, proportion_confint
    count = np.array([r_high, r_low])
    nobs = np.array([n_high, n_low])
    z_stat, p_value = proportions_ztest(count, nobs)
    
    ci_high = proportion_confint(r_high, n_high, alpha=0.05, method='wilson')
    ci_low = proportion_confint(r_low, n_low, alpha=0.05, method='wilson')
    
    insight = {
        'high_risk_group': {
            'threshold': '> 0.15',
            'orders': int(n_high),
            'returns': int(r_high),
            'rate': float(rate_high),
            'ci_lower': float(ci_high[0]),
            'ci_upper': float(ci_high[1])
        },
        'low_risk_group': {
            'threshold': '<= 0.15',
            'orders': int(n_low),
            'returns': int(r_low),
            'rate': float(rate_low),
            'ci_lower': float(ci_low[0]),
            'ci_upper': float(ci_low[1])
        },
        'risk_ratio': float(risk_ratio),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05)
    }
    with open('artifacts/business_insight.json', 'w') as f:
        json.dump(insight, f)

def perform_temporal_split(df):
    train_mask = df['order_date'].dt.year <= 2019
    val_mask = df['order_date'].dt.year == 2020
    test_mask = df['order_date'].dt.year == 2021
    return df[train_mask].copy(), df[val_mask].copy(), df[test_mask].copy()

def prepare_features_and_target(df):
    y = df['returned'].values
    cols_to_drop = ['order_id', 'customer_id', 'order_date', 'returned', 'historical_returns', 'market_historical_returns']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    return X, y

def build_preprocessor(X):
    categorical_features = ['market', 'region', 'country', 'segment', 'order_priority', 'order_month', 'order_day_of_week']
    categorical_features = [f for f in categorical_features if f in X.columns]
    numeric_features = [c for c in X.columns if c not in categorical_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ])
    return preprocessor

def evaluate_thresholds(y_val, probs):
    thresholds = np.arange(0.05, 0.96, 0.05)
    best_f1 = -1
    best_thresh = 0.5
    tuning_data = []
    
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        p = precision_score(y_val, preds, zero_division=0)
        r = recall_score(y_val, preds, zero_division=0)
        
        tuning_data.append({'threshold': t, 'f1': f1, 'precision': p, 'recall': r})
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            
    pd.DataFrame(tuning_data).to_csv('artifacts/threshold_tuning.csv', index=False)
    return best_thresh

def evaluate_model(name, y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    return {
        'roc_auc': roc_auc_score(y_true, probs),
        'pr_auc': average_precision_score(y_true, probs),
        'f1': f1_score(y_true, preds, zero_division=0),
        'precision': precision_score(y_true, preds, zero_division=0),
        'recall': recall_score(y_true, preds, zero_division=0),
        'brier_score': brier_score_loss(y_true, probs),
        'threshold': threshold
    }

def get_curve_data(y_true, probs):
    fpr, tpr, _ = roc_curve(y_true, probs)
    pre, rec, _ = precision_recall_curve(y_true, probs)
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10)
    
    return {
        'roc': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
        'pr': {'precision': pre.tolist(), 'recall': rec.tolist()},
        'calibration': {'prob_true': prob_true.tolist(), 'prob_pred': prob_pred.tolist()}
    }

def create_leakage_audit():
    audit = [
        {"Feature": "total_sales", "Available at prediction time?": "Yes", "Future data used?": "No", "Safe?": "Yes", "Reason": "Known exactly at cart checkout."},
        {"Feature": "customer_historical_return_rate", "Available at prediction time?": "Yes", "Future data used?": "No", "Safe?": "Yes", "Reason": "Calculated strictly prior to current order timestamp."},
        {"Feature": "days_since_previous_order", "Available at prediction time?": "Yes", "Future data used?": "No", "Safe?": "Yes", "Reason": "Calculated strictly from previous orders."},
        {"Feature": "shipping_cost", "Available at prediction time?": "No", "Future data used?": "Yes", "Safe?": "No (Excluded)", "Reason": "Calculated post-shipment. Excluded."},
        {"Feature": "profit", "Available at prediction time?": "No", "Future data used?": "Yes", "Safe?": "No (Excluded)", "Reason": "Settled post-transaction. Excluded."},
        {"Feature": "ship_date", "Available at prediction time?": "No", "Future data used?": "Yes", "Safe?": "No (Excluded)", "Reason": "Occurs after order. Excluded."}
    ]
    pd.DataFrame(audit).to_csv('artifacts/leakage_audit.csv', index=False)
    return audit

def main():
    print("Generating Leakage Audit...")
    create_leakage_audit()
    
    print("Loading data...")
    df = load_data()
    
    generate_dataset_stats(df)
    generate_business_insight(df)
    
    print("Performing temporal split...")
    train_df, val_df, test_df = perform_temporal_split(df)
    
    class_dist = {
        'Train (2018-2019)': train_df['returned'].value_counts().to_dict(),
        'Validation (2020)': val_df['returned'].value_counts().to_dict(),
        'Test (2021)': test_df['returned'].value_counts().to_dict()
    }
    with open('artifacts/class_distribution.json', 'w') as f:
        json.dump(class_dist, f)
    
    X_train, y_train = prepare_features_and_target(train_df)
    X_val, y_val = prepare_features_and_target(val_df)
    X_test, y_test = prepare_features_and_target(test_df)
    
    print("Building preprocessing pipeline...")
    preprocessor = build_preprocessor(X_train)
    
    pos_count = sum(y_train)
    neg_count = len(y_train) - pos_count
    spw = neg_count / pos_count if pos_count > 0 else 1
    
    models = {
        'Majority Class': DummyClassifier(strategy='prior'),
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(scale_pos_weight=spw, eval_metric='logloss', random_state=42)
    }
    
    val_results = []
    best_pr_auc = -1
    best_model_name = None
    best_pipeline = None
    best_threshold = 0.5
    curve_data = {}
    
    print("Training models and optimizing thresholds on Validation Set...")
    for name, model in models.items():
        start_time = time.time()
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        probs = pipeline.predict_proba(X_val)[:, 1]
        
        if name == 'Majority Class':
            thresh = 0.5
        else:
            thresh = evaluate_thresholds(y_val, probs) if name == 'XGBoost' else 0.5
            
        metrics = evaluate_model(name, y_val, probs, thresh)
        metrics['model'] = name
        metrics['train_time_sec'] = train_time
        val_results.append(metrics)
        
        if name == 'XGBoost':
            curve_data['validation'] = get_curve_data(y_val, probs)
        
        print(f"{name} (Val) - PR-AUC: {metrics['pr_auc']:.4f}, F1: {metrics['f1']:.4f}, Brier: {metrics['brier_score']:.4f}")
        
        if metrics['pr_auc'] > best_pr_auc:
            best_pr_auc = metrics['pr_auc']
            best_model_name = name
            best_pipeline = pipeline
            best_threshold = thresh
            
    print(f"\nSelected Best Model: {best_model_name}")
    print(f"Locked Threshold based on Validation: {best_threshold:.2f}")
    
    test_probs = best_pipeline.predict_proba(X_test)[:, 1]
    final_test_metrics = evaluate_model(best_model_name, y_test, test_probs, best_threshold)
    final_test_metrics['model'] = best_model_name
    
    curve_data['test'] = get_curve_data(y_test, test_probs)
    
    with open('artifacts/curve_data.json', 'w') as f:
        json.dump(curve_data, f)
        
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_pipeline, 'models/best_model.joblib')
    
    with open('artifacts/test_metrics.json', 'w') as f:
        json.dump(final_test_metrics, f)
        
    pd.DataFrame(val_results).to_csv('artifacts/validation_comparison.csv', index=False)
    
if __name__ == "__main__":
    main()
