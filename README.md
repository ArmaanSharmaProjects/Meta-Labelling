# Triple Barrier Labeling with Dollar Bars and Meta-Labeling for Financial Event Classification

---

## Overview

This project implements a **full meta-labeling pipeline** for financial time series classification using **dollar bars**, **triple barrier labeling**, and **Random Forest classification**—inspired by *Marcos López de Prado’s Advances in Financial Machine Learning*. The final model predicts whether **crossing moving average events** are worth acting on, enhancing execution with **confidence filtering**.

Built from scratch and applied to **E-mini S&P 500 futures (ES)** 1 minute data, this system applies cutting edge techniques in financial machine learning and backtests model precision under high-confidence trading signals.

---

## Why Dollar Bars?

Traditional time based bars introduce **sampling bias** and **volatility clustering**, obscuring alpha signals.  
Instead, I use **dollar bars**: bars constructed whenever cumulative traded value exceeds a threshold ($2,000,000) to normalize information flow.

### Chosen Threshold: `2,000,000 USD`
This threshold balances signal resolution and noise suppression:
- Higher thresholds reduce overfitting to microstructure noise.
- Lower thresholds increase data points but risk label contamination.

---

## Strategy Signal: Moving Average Crossovers

A basic long/short signal is defined as:
- **Side = +1 (long):** Short-term MA crosses above long-term MA.
- **Side = -1 (short):** Short-term MA crosses below long-term MA.

Feature engineering augments this with:
- **Serial autocorrelation** of returns (persistence signals)
- **Volatility estimates** (risk control)
- **Momentum features** (mean reversion vs. trend following)

---

## Labeling: Triple Barrier Method

To avoid lookahead bias and build a **realistic labeling system**, the **Triple Barrier Method** is applied:
- **Profit-taking barrier (pt)**: `+1 * σ_t`
- **Stop-loss barrier (sl)**: `-2 * σ_t`
- **Vertical barrier (t1)**: 1-day horizon

Barriers are sized using **exp. weighted moving volatility** of 1-day returns:
```python
targets = dailyReturns.ewm(span = 100).std()
```

Only events with targets above `0.002` are retained, ensuring that labels focus on **significant, actionable movements**.

---

## Meta-Labeling Model

A **Random Forest Classifier** is trained to identify which crossover events are likely to succeed:
- Features: volatility, momentum, serial correlation, MA difference
- Labels: Triple barrier bin (1 for success, 0 for failure)

I use a **time-based train/test split** to avoid information leakage.

---

## Confidence Thresholding

Predictions are filtered using **confidence scores (`predict_proba`)**:
- Only trades with **confidence > 75%** are executed
- Ensures **quality over quantity**

---

## Results

Model backtested on high-confidence signals:

| Metric              | Value    |
|---------------------|----------|
| Hit Ratio           | 0.80     |
| Avg Return per Trade| 0.00218  |
| Sharpe Ratio        | 0.59     |

Returns are calculated using real entry/exit prices from `close[t0]` to `close[t1]`, adjusted by predicted side.

---

## File Structure

```bash
.
├── ES_1min_sample.csv
├── tripleBarrierDollarData.csv #Will be created after run
├── main.py #Run
└── README.md
```

---

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*

---
