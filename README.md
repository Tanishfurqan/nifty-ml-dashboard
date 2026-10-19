\# Nifty 50 ML Dashboard



A Streamlit dashboard evaluating whether machine learning models can predict the next-day price direction of 47 Nifty 50 stocks.



\## Models



\- Logistic Regression

\- Random Forest

\- XGBoost

\- LSTM



\## Key Finding



None of the tested models outperformed the always-up baseline in the evaluation. The baseline achieved approximately 51.0% accuracy, compared with approximately 50.5% for the tested models.



\## Features



\- Stock price charts

\- Model performance comparison

\- Backtesting

\- Volatility analysis

\- LSTM evaluation

\- Experiment audit

\- Interactive Try It page using saved models



\## Installation



Install the dashboard and pipeline dependencies:



```bash

pip install -r requirements-pipeline.txt

```



\## Run the Dashboard



```bash

streamlit run app.py

```



\## Disclaimer



This project is for educational and research purposes. Its model outputs are not financial advice or reliable guarantees of future stock performance.



