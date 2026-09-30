from flask import Flask, request, jsonify, render_template
import sqlite3
import pandas as pd
import joblib
import json
import shap
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'superstore.db')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.joblib')
METRICS_PATH = os.path.join(BASE_DIR, 'artifacts', 'test_metrics.json')

# Load Model
model = joblib.load(MODEL_PATH)
with open(METRICS_PATH, 'r') as f:
    metrics = json.load(f)
threshold = metrics['threshold']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/customers', methods=['GET'])
def get_customers():
    conn = sqlite3.connect(DB_PATH)
    customers = pd.read_sql("SELECT DISTINCT customer_id FROM orders LIMIT 200", conn)['customer_id'].tolist()
    conn.close()
    return jsonify(customers)

@app.route('/api/customer-history', methods=['GET'])
def get_customer_history():
    customer_id = request.args.get('customer_id')
    order_date = request.args.get('date')
    
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Historical Stats
    hist_query = f"""
    SELECT 
        COUNT(o.order_id) as hist_orders,
        SUM(CASE WHEN r.order_id IS NOT NULL THEN 1 ELSE 0 END) as hist_returns,
        SUM(o.sales) as total_historical_sales,
        MAX(o.order_date) as last_order_date
    FROM orders o
    LEFT JOIN returns r ON o.order_id = r.order_id
    WHERE o.customer_id = '{customer_id}' AND o.order_date < '{order_date}'
    """
    hist_df = pd.read_sql(hist_query, conn)
    
    historical_orders = int(hist_df['hist_orders'].iloc[0]) if not pd.isna(hist_df['hist_orders'].iloc[0]) else 0
    hist_returns = int(hist_df['hist_returns'].iloc[0]) if not pd.isna(hist_df['hist_returns'].iloc[0]) else 0
    total_sales = float(hist_df['total_historical_sales'].iloc[0]) if not pd.isna(hist_df['total_historical_sales'].iloc[0]) else 0.0
    last_order = hist_df['last_order_date'].iloc[0]
    
    if pd.isna(last_order):
        days_since_prev = 9999
    else:
        days_since_prev = (pd.to_datetime(order_date) - pd.to_datetime(last_order)).days
        
    prior_returns = 1.0
    prior_orders = 7.0
    cust_hist_rate = (hist_returns + prior_returns) / (historical_orders + prior_orders)
    avg_order_value = total_sales / historical_orders if historical_orders > 0 else 0
    
    # 2. Historical Table
    table_query = f"""
    SELECT o.order_date, o.category, o.sales, o.quantity, o.discount,
           CASE WHEN r.order_id IS NOT NULL THEN 'Yes' ELSE 'No' END as returned
    FROM orders o
    LEFT JOIN returns r ON o.order_id = r.order_id
    WHERE o.customer_id = '{customer_id}' AND o.order_date < '{order_date}'
    ORDER BY o.order_date DESC
    LIMIT 10
    """
    table_df = pd.read_sql(table_query, conn)
    conn.close()
    
    return jsonify({
        'historical_orders': historical_orders,
        'hist_returns': hist_returns,
        'cust_hist_rate': cust_hist_rate,
        'avg_order_value': avg_order_value,
        'days_since_prev': days_since_prev,
        'history_table': table_df.to_dict(orient='records')
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    customer_id = data.get('customer_id')
    order_date = data.get('order_date')
    market = data.get('market')
    
    # Construct historical point-in-time features on the backend
    conn = sqlite3.connect(DB_PATH)
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
    
    market_query = f"""
    SELECT 
        COUNT(o.order_id) as mkt_hist_orders,
        SUM(CASE WHEN r.order_id IS NOT NULL THEN 1 ELSE 0 END) as mkt_hist_returns
    FROM orders o
    LEFT JOIN returns r ON o.order_id = r.order_id
    WHERE o.market = '{market}' AND o.order_date < '{order_date}'
    """
    mkt_df = pd.read_sql(market_query, conn)
    conn.close()
    
    historical_orders = int(hist_df['hist_orders'].iloc[0]) if not pd.isna(hist_df['hist_orders'].iloc[0]) else 0
    hist_returns = int(hist_df['hist_returns'].iloc[0]) if not pd.isna(hist_df['hist_returns'].iloc[0]) else 0
    last_order = hist_df['last_order_date'].iloc[0]
    
    mkt_historical_orders = int(mkt_df['mkt_hist_orders'].iloc[0]) if not pd.isna(mkt_df['mkt_hist_orders'].iloc[0]) else 0
    mkt_hist_returns = int(mkt_df['mkt_hist_returns'].iloc[0]) if not pd.isna(mkt_df['mkt_hist_returns'].iloc[0]) else 0
    
    if pd.isna(last_order):
        days_since_prev = 9999
    else:
        days_since_prev = (pd.to_datetime(order_date) - pd.to_datetime(last_order)).days
        
    prior_returns = 1.0
    prior_orders = 7.0
    cust_hist_rate = (hist_returns + prior_returns) / (historical_orders + prior_orders)
    
    mkt_prior_returns = 50.0
    mkt_prior_orders = 300.0
    mkt_hist_rate = (mkt_hist_returns + mkt_prior_returns) / (mkt_historical_orders + mkt_prior_orders)
    
    # Build single-row DataFrame
    input_data = pd.DataFrame([{
        'market': data['market'],
        'region': data['region'],
        'country': data['country'],
        'segment': data['segment'],
        'order_priority': data['order_priority'],
        'total_sales': float(data['total_sales']),
        'total_quantity': int(data['total_quantity']),
        'average_discount': float(data['average_discount']),
        'max_discount': float(data['max_discount']),
        'number_of_products': int(data['number_of_products']),
        'number_of_categories': int(data['number_of_categories']),
        'historical_orders': historical_orders,
        'days_since_previous_order': days_since_prev,
        'customer_historical_return_rate': cust_hist_rate,
        'market_historical_orders': mkt_historical_orders,
        'market_historical_return_rate': mkt_hist_rate,
        'order_month': pd.to_datetime(data['order_date']).month,
        'order_day_of_week': pd.to_datetime(data['order_date']).dayofweek,
        'is_weekend': 1 if pd.to_datetime(data['order_date']).dayofweek in [5, 6] else 0
    }])
    
    prob = model.predict_proba(input_data)[0][1]
    
    # SHAP logic
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']
    input_processed = preprocessor.transform(input_data)
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(input_processed)
    
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
    num_features = preprocessor.transformers_[0][2]
    feature_names = list(num_features) + list(cat_features)
    
    shap_df = pd.DataFrame({'Feature': feature_names, 'Contribution': shap_values[0]})
    shap_df['Abs'] = shap_df['Contribution'].abs()
    shap_df = shap_df.sort_values('Abs', ascending=False).head(5)
    
    shap_list = []
    for _, row in shap_df.iterrows():
        shap_list.append({
            'feature': row['Feature'],
            'contribution': float(row['Contribution']),
            'direction': 'increase' if row['Contribution'] > 0 else 'decrease'
        })
        
    return jsonify({
        'probability': float(prob),
        'threshold': float(threshold),
        'classification': 'HIGH RISK' if prob >= threshold else 'LOW RISK',
        'shap': shap_list
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
