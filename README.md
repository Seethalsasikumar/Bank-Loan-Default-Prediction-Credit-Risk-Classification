# Bank Loan Default Prediction - Credit Risk Classification

📌 Overview

This project builds a data-driven credit scoring model to predict whether a loan applicant is likely to default (BAD) or repay (GOOD), helping a retail bank reduce default-related losses while continuing to approve creditworthy customers.

🎯 Objective

- Predict loan default risk at the application stage using historical customer data
- Meet a business-defined recall target: correctly identify at least 85% of defaulters before loan disbursement
- Translate model outputs into a practical, tiered credit approval policy

📊 Dataset

Source: Historical loan application data (HMEQ-style dataset), including applicant financials, credit history, and loan purpose 

Target variable: BAD (0 = Paid, 1 = Default)

Key features: DEBTINC (debt-to-income ratio), DEROG (derogatory reports), DELINQ (delinquent credit lines), CLAGE (credit line age), NINQ (recent credit inquiries), REASON, JOB

🛠️ Tools & Libraries

- Language: Python
- Libraries: pandas, NumPy, scikit-learn (Random Forest, Logistic Regression, RandomizedSearchCV, StratifiedKFold), Matplotlib/Seaborn
- Techniques: Median/zero imputation, one-hot encoding, stratified train-test split, hyperparameter tuning, threshold optimisation, precision-recall analysis, feature importance (Random Forest)

🔍 Approach

1. EDA — analysed categorical (REASON, JOB) and numerical features against default outcomes; identified DEBTINC, DEROG, DELINQ, NINQ, and YOJ as the strongest predictors
   
2. Data cleaning — median imputation for skewed continuous variables, zero-imputation for count-based fields, one-hot encoding for categoricals

3. Modelling — trained a Random Forest Classifier as the primary model, benchmarked against Logistic Regression as a transparent baseline

4. Threshold tuning — since recall (catching defaulters) mattered more than precision for the bank, the classification threshold was lowered from 0.5 → 0.36 to meet the 85% recall target

5. Business translation — mapped model probability outputs into a tiered credit approval policy (auto-reject, manual review, auto-approve bands)


📈 Key Results

- Random Forest outperformed Logistic Regression across every metric, with a notably stronger ROC AUC (0.950 vs 0.784)
- Lowering the decision threshold to 0.36 improved recall by 13.9 percentage points (71.7% → 85.7%), meeting the bank's risk target with only a modest precision trade-off
- DEBTINC was the single strongest predictor of default, consistent with standard credit risk practice, followed by CLAGE, DELINQ, and MORTDUE

💡 Business Recommendation

Deploy Random Forest at a 0.36 threshold as the primary decision engine, with Logistic Regression retained for governance and explainability (compliance audits, edge-case human review). Model outputs feed into a tiered policy: auto-reject above 60% default probability, manual review for the 36–60% band, and streamlined approval below that.

