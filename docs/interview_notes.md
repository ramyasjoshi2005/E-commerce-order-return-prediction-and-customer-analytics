# Interview notes (short)

Numbers come from `artifacts/`. Re-run the pipeline if you retrain.

1. **Problem:** Predict return at order placement; identify associated factors.
2. **Why placement?** Later features (ship cost, profit) arrive after cost is incurred.
3. **Leakage:** Dropped duplicate order_ids; history uses only earlier orders; 2021 held out.
4. **Dataset choice:** Global Superstore + returns.csv have 0 Order ID overlap — did not fake labels. Modeling data is US Superstore 2018–2021, 5,009 orders, 5.91% returns. No India in labeled data.
5. **Stats:** Two-proportion z-test on prior-return groups. z=0.54, p=0.59. Do not reject H0. Not causation.
6. **Models:** Baseline, LR, RF, XGBoost. Selected LR on 2020 PR-AUC (0.148), not because it is fashionable.
7. **Metrics:** Test PR-AUC 0.154, ROC-AUC 0.693, F1 0.232 at threshold 0.65. Accuracy is a poor headline (~94% by predicting no return).
8. **Threshold:** Max F1 on validation; locked; never tuned on test.
9. **Calibration:** Brier 0.17; class-weighted LR overstates probabilities. Treat as a ranking score.
10. **SHAP:** Feature contribution to the score, not a cause.
11. **Business:** West region has a much higher observed return rate. Prior-return gap is small and not significant. Recommend review queues + A/B tests, not auto-blocking, not a claimed X% return reduction.
