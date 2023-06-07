# Databricks notebook source
# MAGIC %md
# MAGIC # Machine Learning Assurance

# COMMAND ----------

# MAGIC %pip install scikit-learn forestci

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Machine Learning

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC Tasks for cell below:
# MAGIC - what health datasets are available in sklearn for regression
# MAGIC 
# MAGIC - `X_clf` is the inputs to the classifcition (clf) model.

# COMMAND ----------

from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_diabetes

# input (X) and output (y) data
X_clf, y_clf = load_breast_cancer(return_X_y=True, as_frame=True) # classification dataset (clf)
X_reg, y_reg = load_diabetes(return_X_y=True, as_frame=True) # regression dataset (reg)

# COMMAND ----------

#X_clf.describe()
X_reg.describe()

# COMMAND ----------

import seaborn as sns

sns.histplot(X_reg['sex'])

# COMMAND ----------

# MAGIC %md
# MAGIC Tasks for cell below:
# MAGIC - How many rows and columns are there in the dataset?
# MAGIC - Are there any missing values?

# COMMAND ----------

# MAGIC %md
# MAGIC Tasks for cell below:
# MAGIC - what is the proportion of train/test?
# MAGIC - How can you ensure that the train and test split are always the same?

# COMMAND ----------

from sklearn.model_selection import train_test_split

# training and testing data
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg)

# COMMAND ----------

# MAGIC %md
# MAGIC Tasks for cell below:
# MAGIC - what other types of models are available?
# MAGIC - what model would you use for regression?

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

# untrained model
model_clf = RandomForestClassifier()
model_reg = RandomForestRegressor()

# COMMAND ----------

# fit model on training data
model_clf.fit(X_train_clf, y_train_clf)
model_reg.fit(X_train_reg, y_train_reg)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Performance

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classification

# COMMAND ----------

import pandas as pd
from sklearn.metrics import classification_report

y_pred_clf = model_clf.predict(X_test_clf) # getting predictions
pd.DataFrame(classification_report(y_test_clf, y_pred_clf, output_dict=True, target_names=["malignant", "benign"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Regression

# COMMAND ----------

from sklearn.metrics import mean_absolute_percentage_error

y_pred_reg = model_reg.predict(X_test_reg) # getting predictions
mape = mean_absolute_percentage_error(y_test_reg, y_pred_reg)
print(f"Mean Absolute Percentage Error: {mape:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fairness 

# COMMAND ----------

X_test_reg_male = X_test_reg[X_test_reg['sex'] < 0]
X_test_reg_female = X_test_reg[X_test_reg['sex'] >= 0]

y_test_reg_male = y_test_reg[X_test_reg['sex'] < 0]
y_test_reg_female = y_test_reg[X_test_reg['sex'] >= 0]

y_pred_reg_male = model_reg.predict(X_test_reg_male)
y_pred_reg_female = model_reg.predict(X_test_reg_female)

mape_male = mean_absolute_percentage_error(y_test_reg_male, y_pred_reg_male)
mape_female = mean_absolute_percentage_error(y_test_reg_female, y_pred_reg_female)

print(f"Mean Absolute Percentage Error for Males: {mape_male:.2f}")
print(f"Mean Absolute Percentage Error for Females: {mape_female:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calibration (Classification)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Brier Skill Score

# COMMAND ----------

from sklearn.metrics import brier_score_loss

bs = brier_score_loss(y_test_clf, y_pred_clf)
print(f"Brier Score: {bs:.2f}")

# COMMAND ----------

from sklearn.dummy import DummyClassifier

# fit baseline model on training data
model_clf_bl = DummyClassifier()
model_clf_bl.fit(X_train_clf, y_train_clf)
y_pred_clf_bl = model_clf_bl.predict(X_test_clf)
bs_bl = brier_score_loss(y_test_clf, y_pred_clf_bl)
print(f"Baseline Brier Score: {bs_bl:.2f}")

# COMMAND ----------

# zero is baseline and one is perfect
print("Brier Skill Score: ", 1 - bs / bs_bl)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calibration Plot

# COMMAND ----------

from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibrationDisplay

y_prob_clf = model_clf.predict_proba(X_test_clf)[:,1]
prob_true_clf, prob_pred_clf = calibration_curve(y_test_clf, y_prob_clf, n_bins=5)
disp = CalibrationDisplay(prob_true_clf, prob_pred_clf, y_pred_clf)
disp.plot()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Uncertainty (Regression)

# COMMAND ----------

import forestci as fci
import numpy as np

# prediction variances
y_pred_reg_var = fci.random_forest_error(model_reg, X_train_reg, X_test_reg)

lower = y_pred_reg - 2 * np.sqrt(y_pred_reg_var)
upper = y_pred_reg + 2 * np.sqrt(y_pred_reg_var)

conf_df = pd.DataFrame({'lower': lower, 'upper': upper, 'truth': y_test_reg})
conf_df

# COMMAND ----------

from matplotlib import pyplot as plt

plt.errorbar(y_test_reg, y_pred_reg, yerr=2*np.sqrt(y_pred_reg_var), fmt='o')
plt.plot([0, 350], [0, 350], 'k--')
plt.xlabel('Truth')
plt.ylabel('Predicted ')
plt.show()

# COMMAND ----------

results = ((conf_df.truth > conf_df.lower) & (conf_df.truth < conf_df.upper)).value_counts(normalize=True)
print(f"Coverage: {results[1] * 100:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Explainable

# COMMAND ----------

# MAGIC %md
# MAGIC ### Permutation Feature Importance

# COMMAND ----------

from sklearn.inspection import permutation_importance
import seaborn as sns

result = permutation_importance(
    model_clf, X_test_clf, y_test_clf, n_repeats=10, scoring="precision"
)

importances_df = pd.DataFrame({"Feature": X_test_clf.columns, "Importance": result.importances_mean})\
    .sort_values("Importance", ascending=False)\
    .head(10)
sns.barplot(x="Importance", y="Feature", data=importances_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### SHAP

# COMMAND ----------

import shap

explainer = shap.Explainer(model_clf)
shap_values = explainer(X_test_clf)
explaination = shap.Explanation(shap_values[:,:,1], feature_names=X_train_clf.columns)

patient = 5
shap.plots.waterfall(explaination[patient])
