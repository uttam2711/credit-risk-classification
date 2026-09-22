# Financial Credit Risk Classification

Predicts whether a loan applicant will default, using real financial data. The tricky part: only 6.7% of people in the data actually default — so the model has to learn from a very small number of "bad" examples compared to "good" ones.

**Dataset:** [Give Me Some Credit (Kaggle)](https://www.kaggle.com/c/GiveMeSomeCredit/data) — 150,000 loan applicants, 10 financial features.

## What I did

1. Cleaned messy data — fixed impossible values (like a debt ratio of 300,000%) and placeholder error codes hiding in the "late payment" columns.
2. Combined three separate "how many times were they late" columns into one clean feature.
3. Used class weighting so the model doesn't just ignore the rare "default" cases.
4. Compared Logistic Regression vs Random Forest using 5-fold cross-validation, tuned with GridSearchCV.

## Two things I found that I didn't expect

**1. I expected multicollinearity, but didn't find any — until I created it myself.**
I checked for multicollinearity (VIF) between the three late-payment columns, expecting them to overlap. They didn't. So instead I combined them into one new "total late payments" feature — and *that* new feature turned out to be fully redundant with the three originals (which makes sense, since it's just their sum). I dropped the three originals and kept the one combined feature.

**2. My first Random Forest model looked fine but was actually bad at the one thing that mattered.**
The first version had 93% accuracy — sounds great, but it only caught 14% of actual defaulters. It was quietly playing it safe by rarely flagging anyone as risky. After tuning (mainly using shallower trees instead of deep ones), the same model started catching 82% of real defaulters. Accuracy is a misleading number here — recall on defaulters is what actually matters for a bank.

## Final model

Random Forest (tuned), scoring:
- Recall on defaults: 0.82
- ROC-AUC: 0.861

## Tools

Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Statsmodels (for VIF)

## Files

- `main.ipynb` — full analysis and modeling
- `credit_risk_model.pkl` — final trained model
