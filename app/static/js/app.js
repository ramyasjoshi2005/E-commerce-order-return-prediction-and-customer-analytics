let currentCustomerStats = null;

document.addEventListener("DOMContentLoaded", () => {
    fetch('/api/customers')
        .then(res => res.json())
        .then(data => {
            const select = document.getElementById('customer_id');
            data.forEach(id => {
                const opt = document.createElement('option');
                opt.value = id;
                opt.textContent = id;
                select.appendChild(opt);
            });
        });
});

function fetchCustomerHistory() {
    const customer_id = document.getElementById('customer_id').value;
    const order_date = document.getElementById('order_date').value;
    
    if (!customer_id || !order_date) return;
    
    fetch(`/api/customer-history?customer_id=${customer_id}&date=${order_date}`)
        .then(res => res.json())
        .then(data => {
            currentCustomerStats = data;
            document.getElementById('customer_profile').style.display = 'block';
            
            document.getElementById('hist_orders').textContent = data.historical_orders;
            document.getElementById('hist_returns').textContent = data.hist_returns;
            document.getElementById('hist_rate').textContent = (data.cust_hist_rate * 100).toFixed(1) + '%';
            document.getElementById('avg_aov').textContent = '$' + data.avg_order_value.toFixed(2);
            document.getElementById('days_prev').textContent = data.days_since_prev === 9999 ? 'N/A' : data.days_since_prev;
            
            const tbody = document.querySelector('#history_table tbody');
            tbody.innerHTML = '';
            data.history_table.forEach(row => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${row.order_date.split(' ')[0]}</td>
                    <td>${row.category}</td>
                    <td>$${row.sales.toFixed(2)}</td>
                    <td>${row.quantity}</td>
                    <td>${row.discount}</td>
                    <td>${row.returned}</td>
                `;
                tbody.appendChild(tr);
            });
        });
}

function predictRisk() {
    const customer_id = document.getElementById('customer_id').value;
    if (!customer_id) {
        alert("Please select a customer first.");
        return;
    }
    
    const payload = {
        customer_id: customer_id,
        market: document.getElementById('market').value,
        region: document.getElementById('region').value,
        country: document.getElementById('country').value,
        segment: document.getElementById('segment').value,
        order_priority: document.getElementById('order_priority').value,
        total_sales: document.getElementById('total_sales').value,
        total_quantity: document.getElementById('total_quantity').value,
        average_discount: document.getElementById('average_discount').value,
        max_discount: document.getElementById('max_discount').value,
        number_of_products: document.getElementById('number_of_products').value,
        number_of_categories: document.getElementById('number_of_categories').value,
        order_date: document.getElementById('order_date').value
    };
    
    console.log("Selected customer: " + payload.customer_id);
    console.log("Order date: " + payload.order_date);
    console.log("Prediction payload: ", payload);
    
    fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById('result_section').style.display = 'block';
        
        const probElem = document.getElementById('res_prob');
        const classElem = document.getElementById('res_class');
        
        const probPct = (data.probability * 100).toFixed(1);
        probElem.textContent = probPct + '%';
        classElem.textContent = data.classification;
        
        if (data.classification === 'HIGH RISK') {
            probElem.style.color = '#c40000';
            classElem.style.color = '#c40000';
        } else {
            probElem.style.color = '#007185';
            classElem.style.color = '#007185';
        }
        
        document.getElementById('progress_fill').style.width = probPct + '%';
        document.getElementById('res_thresh').textContent = (data.threshold * 100).toFixed(0);
        
        // Render SHAP
        const shapContainer = document.getElementById('shap_container');
        shapContainer.innerHTML = '';
        
        const maxAbs = Math.max(...data.shap.map(s => Math.abs(s.contribution)));
        
        data.shap.forEach(s => {
            const item = document.createElement('div');
            item.className = 'shap-item';
            
            const width = (Math.abs(s.contribution) / maxAbs) * 100;
            const barClass = s.direction === 'increase' ? 'shap-increase' : 'shap-decrease';
            const arrow = s.direction === 'increase' ? '↑' : '↓';
            
            item.innerHTML = `
                <div class="shap-label">${s.feature}</div>
                <div class="shap-bar-container">
                    <div class="shap-bar ${barClass}" style="width: ${width}%">${arrow}</div>
                </div>
            `;
            shapContainer.appendChild(item);
        });
        
        // Scroll to result
        document.getElementById('result_section').scrollIntoView({ behavior: 'smooth' });
    });
}
