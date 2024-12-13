
# Metric summaries

## Manual approach - Phase I

### Engagement-level dataset
Initially, using a single train-test split, the Random Forest model achieved F1 scores of 60% and 92% for the cheater and non-cheater classes respectively. However, to obtain a more robust estimate of model performance, 
5-fold stratified cross validation was implemented - one version with smote (on the training set in each fold) and one without, to quantify its impact. 

For the negative (not cheater) class, here are the average metrics (in percent) across all 5 folds:


| Sampling Strategy | Precision | Recall | F1 Score |
| -------- | ------- | -------- | ------- |
| Without SMOTE | 83.33 ± 0.04 | 99.6 ± 0.3 | 90.8 ± 0.1 |
| With SMOTE | 83.5 ± 0.1 | 79.9 ± 0.7 | 81.6 ± 0.4 |


For the positive (cheater) class, here are the average metrics across all 5 folds:

| Sampling Strategy | Precision | Recall | F1 Score |
| -------- | ------- | -------- | ------- |
| Without SMOTE | 22.0 ± 10.0 | 0.3 ± 0.1 | 0.6 ± 0.2 |
| With SMOTE | 17.4 ± 0.5 | 21.1 ± 0.3 | 19.1 ± 0.3 |

The average accuracies with and without SMOTE were 70.1 ± 0.6 and 83.1 ± 0.2 respectively.


### Player-level dataset

Initially, using a single train-test split, the Random Forest model achieved F1 scores of 17% and 84% for the cheater and non-cheater classes respectively. 

However, I followed up with the same model set-up - Random Forest classifier with 5-fold stratified cross validation. 

For the negative (not cheater) class, here are the average metrics (in percent) across all 5 folds:

| Sampling Strategy | Precision | Recall | F1 Score |
| -------- | ------- | -------- | ------- |
| Without SMOTE | 83.38 ± 0.05 | 99.7 ± 0.3 | 90.8 ± 0.1 |
| With SMOTE | 83.5 ± 0.3 | 83.4 ± 1.3 | 83.5 ± 0.6 |

For the positive (cheater) class, here are the average metrics across all 5 folds:

| Sampling Strategy | Precision | Recall | F1 Score |
| -------- | ------- | -------- | ------- |
| Without SMOTE | 39.2 ± 15.7 | 0.7 ± 0.2 | 1.3 ± 0.3 |
| With SMOTE | 17.6 ± 1.8 | 17.7 ± 2.5 | 17.6 ± 2.1 |

The average accuracies with and without SMOTE were 72.5 ± 0.9 and 83.2 ± 0.3 respectively.

### Isolation forest

With `contamination = auto`, regardless of which dataset the model was trained on, both performed poorly (AUCs around 0.5). 

## Manual approach - Phase II

### XGBoost

* `scale_pos_weight=5` to address class imbalance
* `learning_rate=0.1` and `max_depth=6` to control overfitting
* `reg_alpha=0.5` and `reg_lambda=1` for L1 and L2 regularization
* `subsample=0.8` and `colsample_bytree=0.8` to reduce overfitting
* `n_estimators=100` and `tree_method='hist'` for computational efficiency
* `max_delta_step=1` to prevent aggressive updates favoring the majority class
* f1 custom metric and `num_boost_round=50`

Here are the metrics for both classes (in percent):

| Class | Precision | Recall | F1 Score |
| -------- | ------- | -------- | ------- |
| Not cheater | 89 | 70 | 78 |
| Cheater | 27 | 55 | 36 |

The accuracy of this was model was 68% with an AUC of 0.69.

### Isolation Forest

AUC of 0.50

## AI approach - Phase I

For the positive class the 5-fold cross-validation results for Random Forest showed:

| Class | Precision | Recall | F1 Score |
| -------- | ------- | -------- | ------- |
| Not cheater | 96.8 ± 0.1 | 99.8 ± 0.1 | 98.2 ± 0.1 |
| Cheater | 28.4 ± 11.0 | 2.8 ± 1.9 | 5.0 ± 3.3 |

The average accuracy was 96.5 ± 0.2.

Similarly, the LSTM model performance was also subpar:

| Class | Precision | Recall | F1 Score |
| -------- | ------- | -------- | ------- |
| Not cheater | 97 | 76 | 85 |
| Cheater | 3 | 24 | 6 |

The accuracy was 74%.

## AI approach - Phase II

All results below are for the positive class.

Logistic Regression Results:

accuracy: 0.5913
precision: 0.1878
recall: 0.4368
f1: 0.2627

Random Forest (3 features) Results:
accuracy: 0.5689
precision: 0.1884
recall: 0.4799
f1: 0.2706

Random Forest (all features) Results:
accuracy: 0.6815
precision: 0.2497
recall: 0.4544
f1: 0.3223