## 1. Methodology
```
┌──────────────────────┐
│ Dataset Collection   │
└─────────┬────────────┘
          ↓
┌────────────────────────────┐
│ Data Pre-processing        │
│ & Feature Preparation      │
└─────────┬──────────────────┘
          ↓
┌────────────────────────────┐
│ Sampling Techniques        │
│ (5 Resampling Methods)     │
└─────────┬──────────────────┘
          ↓
┌────────────────────────────┐
│ Model Training             │
│ (4 ML Classifiers)         │
└─────────┬──────────────────┘
          ↓
┌────────────────────────────┐
│ Model Evaluation           │
│ (Accuracy, F1, ROC-AUC)    │
└─────────┬──────────────────┘
          ↓
┌────────────────────────────┐
│ Comparative Analysis       │
│ & Best Technique Selection │
└────────────────────────────┘
```
The methodology follows a structured experimental pipeline where multiple sampling techniques are applied to handle class imbalance, and their impact on different machine learning models is evaluated under identical conditions.

The final comparison identifies the most effective sampling–model combination based on empirical performance.

## 2. Description

* Task Type: Binary Classification (Imbalanced Dataset)

* Dataset Used: Credit Card Fraud Detection Dataset

* Problem Nature: Extreme class imbalance

* Objective: Evaluate how different resampling techniques affect model performance

Dataset Details

* Majority Class: Legitimate Transactions (Class 0)

* Minority Class: Fraudulent Transactions (Class 1)

Models Evaluated

* Logistic Regression

* Support Vector Machine (SVM)

* Random Forest

* Gradient Boosting

* Sampling Techniques Applied

* No Sampling (Baseline)

* Random Under Sampling

* Random Over Sampling

* SMOTE (Synthetic Minority Oversampling Technique)

* NearMiss

* Execution Environment

* Python

* Scikit-learn

* Imbalanced-learn

* Google Colab / Local Machine

## 3. Input / Output
Input

* Tabular transaction data

* Numerical features extracted from financial transactions

Example:
```bash
V1 = -1.3598, V2 = -0.0727, ..., Amount = 149.62
```
Output

* Predicted class label:

    * 0 → Legitimate Transaction

    * 1 → Fraudulent Transaction

Model Comparison Output

* Accuracy

* F1-score

* ROC-AUC

* Comparative accuracy matrix across sampling techniques

## 4. Results Summary

* Tree-based models significantly outperform linear models on imbalanced data

* Sampling techniques improve minority class detection

* NearMiss and aggressive under-sampling lead to loss of critical information

* Oversampling techniques combined with ensemble models achieve the best results

| Model           | No Sampling | Random Under | Random Over | SMOTE     | NearMiss |
| --------------- | ----------- | ------------ | ----------- | --------- | -------- |
| Logistic Reg.   | 98.71       | 60.00        | 90.97       | 92.26     | 29.03    |
| SVM             | 94.84       | 52.26        | 95.48       | 96.13     | 30.97    |
| Random Forest   | 98.71       | 63.87        | **99.35**   | **99.35** | 38.71    |
| Gradient Boost. | 98.71       | 35.48        | **99.35**   | **99.35** | 14.19    |

Highest Accuracy Achieved By:

* Random Forest + SMOTE / Random Over-Sampling

* Gradient Boosting + SMOTE / Random Over-Sampling

## 5. Key Observations

* High accuracy without sampling is misleading due to poor minority class detection

* Linear models fail to capture complex fraud patterns even after resampling

* Tree-based ensemble models benefit the most from oversampling techniques

* Random Under Sampling and NearMiss degrade performance due to information loss

## 6. Conclusion

This project demonstrates that handling class imbalance is critical for building reliable classification models.

Key takeaways:

* Accuracy alone is not sufficient for evaluating imbalanced datasets

* Sampling techniques must be chosen carefully based on data size and model type

* Ensemble learning combined with oversampling provides the best trade-off between performance and robustness

* The optimal approach depends on application requirements rather than a single evaluation metric

This framework is applicable to real-world domains such as fraud detection, medical diagnosis, and anomaly detection, where minority class identification is crucial.