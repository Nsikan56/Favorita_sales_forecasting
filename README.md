# 🛒 Favorita Grocery Sales Forecasting

End-to-end time series forecasting project using XGBoost, SHAP, and real-world grocery data.  
I don’t just tune models — I explain *why* they behave the way they do.

---

## 🚀 Overview

This project predicts daily item-level sales for a major grocery chain using structured time series data.  
It incorporates calendar events, oil prices, holidays, store metadata, and engineered time features.

> ⚠️ Note: Raw data files are not included in this repository due to size.
> You can download the dataset from [Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data).


- 📆 Data: Favorita (Ecuador), 2013–2017
- 🤖 Model: XGBoost Regressor (tuned with GridSearchCV)
- 🧠 Explainability: SHAP (summary + local waterfall)
- 📈 Metrics: RMSE on holdout set

---

## 🧠 Project Highlights

| Step | Highlights |
|------|------------|
| 📊 EDA | Missing value handling, trend visualization |
| 🔨 Feature Engineering | Lag variables, rolling means, promo flags, holiday joins |
| 🧪 Model Tuning | GridSearchCV (light version on 10k rows) |
| 📉 Evaluation | RMSE, prediction plot, scatter vs actual |
| 🔍 Explainability | SHAP summary + waterfall plots |
| 💾 Output | Saved `.pkl` model, visualizations |

---

## 📈 Model Performance

| Metric       | Score    |
|--------------|----------|
| RMSE (val)   | 478.85   |

### 📊 Actual vs Predicted Plot
![Actual vs Predicted](actual_vs_predicted.png)

### 📊 Predicted vs Actual Scatter
![Predicted vs Actual](predicted_vs_actual.png)

---

## 🔍 Feature Importance

### 📌 SHAP Summary Plot
Shows which features most influenced the model.

![SHAP Summary](shap_summary.png)

### 🧠 SHAP Local Waterfall
Detailed explanation for a single prediction.

![SHAP Waterfall](shap_waterfall.png)

### 🔢 XGBoost Top 20 Features
Built-in feature importances based on split frequency.

![XGBoost Feature Importance](xgb_feature_importance.png)

---

## 📂 Project Structure

```
favorita_sales_forecasting/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── stores.csv
│   ├── oil.csv
│   ├── holidays_events.csv
│   ├── transactions.csv
│   └── sample_submission.csv
│
├── images/
│   ├── shap_summary.png
│   ├── shap_waterfall.png
│   ├── actual_vs_predicted.png
│   ├── predicted_vs_actual.png
│   └── xgb_feature_importance.png
│
├── notebooks/
│   └── 01_EDA.ipynb
├── models/
│   └── xgb_favorita_model.pkl
├── requirements.txt
└── README.md
```

---

## 💡 Key Tools Used

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-orange)
![Pandas](https://img.shields.io/badge/Pandas-1.4-lightgrey)
![SHAP](https://img.shields.io/badge/SHAP-black)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

---

## 📌 Future Improvements

- Add LSTM/Prophet comparison
- Use Optuna for deeper hyperparameter search
- Build Streamlit forecast explorer

---

## 📬 Author

**Nsikan**  
📧 [Email me](mailto:nsikanumoh56@gmail.com)  
🌐 Portfolio: [nsikan-portfolio.framer.website](https://nsikan-portfolio.framer.website)

---