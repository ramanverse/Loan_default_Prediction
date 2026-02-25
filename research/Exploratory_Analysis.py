import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data locally
try:
    df = pd.read_csv('../application_train.csv')
except FileNotFoundError:
    df = pd.read_csv('application_train.csv')

df.head()

import numpy as np

print("Initial shape:", df.shape)
print("Columns:", len(df.columns))
print("Target distribution:")
if 'TARGET' in df.columns:
    print(df['TARGET'].value_counts(normalize=True))
else:
    print("No TARGET column found.")

missing = df.isnull().mean().sort_values(ascending=False)
display(missing.head(20))

df_eda = df.sample(n=10000, random_state=42) if len(df) > 10000 else df.copy()
print("df_eda.shape:", df_eda.shape)

cols_to_drop = missing[missing > 0.40].index.tolist()
df_clean = df.drop(columns=cols_to_drop)
print("Dropped columns:", len(cols_to_drop))
print("Shape after drop:", df_clean.shape)


if 'DAYS_EMPLOYED' in df_clean.columns:
    df_clean['DAYS_EMPLOYED'] = df_clean['DAYS_EMPLOYED'].replace(365243, np.nan)


if 'DAYS_BIRTH' in df_clean.columns and 'AGE' not in df_clean.columns:
    df_clean['AGE'] = -df_clean['DAYS_BIRTH'] / 365.0

if 'DAYS_EMPLOYED' in df_clean.columns and 'EMPLOY_YEARS' not in df_clean.columns:
    df_clean['EMPLOY_YEARS'] = -(df_clean['DAYS_EMPLOYED'] / 365.0)

if {'AMT_CREDIT','AMT_INCOME_TOTAL'}.issubset(df_clean.columns) and 'CREDIT_INCOME_RATIO' not in df_clean.columns:
    df_clean['CREDIT_INCOME_RATIO'] = df_clean['AMT_CREDIT'] / df_clean['AMT_INCOME_TOTAL'].replace({0: np.nan})

if {'AMT_ANNUITY','AMT_INCOME_TOTAL'}.issubset(df_clean.columns) and 'ANNUITY_INCOME_RATIO' not in df_clean.columns:
    df_clean['ANNUITY_INCOME_RATIO'] = df_clean['AMT_ANNUITY'] / df_clean['AMT_INCOME_TOTAL'].replace({0: np.nan})

exts = [c for c in ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3'] if c in df_clean.columns]
if exts and 'EXT_SOURCES_MEAN' not in df_clean.columns:
    df_clean['EXT_SOURCES_MEAN'] = df_clean[exts].mean(axis=1)


print("Nulls remaining (total):", df_clean.isnull().sum().sum())
print("Sample of engineered features:")
display(df_clean[['AGE','EMPLOY_YEARS','CREDIT_INCOME_RATIO','ANNUITY_INCOME_RATIO','EXT_SOURCES_MEAN']].head())

from sklearn.impute import SimpleImputer

df_model = df_clean.copy()


num_cols = df_model.select_dtypes(include=['int64','float64']).columns.tolist()
num_cols = [c for c in num_cols if c not in ('SK_ID_CURR',)]
num_imputer = SimpleImputer(strategy='median')
df_model[num_cols] = num_imputer.fit_transform(df_model[num_cols])


cat_cols = df_model.select_dtypes(include=['object']).columns.tolist()
for c in cat_cols:
    df_model[c] = df_model[c].fillna('MISSING')
    df_model[c] = pd.Categorical(df_model[c]).codes


print("Nulls in df_model (total):", df_model.isnull().sum().sum())
print("Model-ready shape:", df_model.shape)

def cap_iqr(series, k=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return series.clip(lower=lower, upper=upper)


cap_features = ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','CREDIT_INCOME_RATIO']
cap_features = [c for c in cap_features if c in df_model.columns]
for c in cap_features:
    df_model[c] = cap_iqr(df_model[c], k=1.5)

print("Applied IQR capping to:", cap_features)



df['TARGET'].value_counts(normalize=True)







df_eda = df.sample(10000, random_state=42)
df_eda.shape

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
sns.histplot(df_eda['AMT_INCOME_TOTAL'], kde=True)
plt.title("Income Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df_eda['AMT_CREDIT'], kde=True, color='green')
plt.title("Credit Amount Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x=df_eda['TARGET'], y=df_eda['AMT_INCOME_TOTAL'])
plt.title("Income vs Default")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(
    x=df_eda['AMT_INCOME_TOTAL'],
    y=df_eda['AMT_CREDIT'],
    alpha=0.3
)
plt.title("Income vs Credit")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(
    df_eda[['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','CNT_CHILDREN']].corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(7,5))
sns.kdeplot(df_eda['EXT_SOURCE_2'], fill=True, label='EXT_SOURCE_2')
sns.kdeplot(df_eda['EXT_SOURCE_3'], fill=True, label='EXT_SOURCE_3')
plt.title("Distribution of External Credit Scores")
plt.xlabel("Score (0 to 1)")
plt.legend()
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x=df_eda['TARGET'], y=df_eda['EXT_SOURCE_2'])
plt.title("EXT_SOURCE_2 vs Default")
plt.xlabel("Default (1 = Yes)")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x=df_eda['TARGET'], y=df_eda['EXT_SOURCE_3'])
plt.title("EXT_SOURCE_3 vs Default")
plt.xlabel("Default (1 = Yes)")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x=df_eda['TARGET'], y=df_eda['EXT_SOURCE_3'])
plt.title("EXT_SOURCE_3 vs Default")
plt.xlabel("Default (1 = Yes)")
plt.show()

df_eda['AGE'] = -df_eda['DAYS_BIRTH'] / 365
plt.figure(figsize=(6,4))
sns.histplot(df_eda['AGE'], kde=True, color='purple')
plt.title("Age Distribution of Applicants")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x=df_eda['TARGET'], y=df_eda['AGE'])
plt.title("Age vs Default")
plt.show()

import numpy as np

df_eda['EMPLOY_YEARS'] = df_eda['DAYS_EMPLOYED'].replace(365243, np.nan) * -1 / 365
plt.figure(figsize=(6,4))
sns.histplot(df_eda['EMPLOY_YEARS'], kde=True, color='teal')
plt.title("Employment Length Distribution")
plt.show()

df_eda.describe()

df_eda['TARGET'].value_counts(normalize=True)

plt.figure(figsize=(5,4))
sns.countplot(x=df_eda['TARGET'], palette='pastel')
plt.title("Target Variable Distribution")
plt.xlabel("TARGET (0 = Non-Default, 1 = Default)")
plt.ylabel("Count")
plt.show()

df_eda.groupby('TARGET')['AMT_INCOME_TOTAL'].mean()

plt.figure(figsize=(6,4))
sns.barplot(
    x=df_eda.groupby('TARGET')['AMT_INCOME_TOTAL'].mean().index,
    y=df_eda.groupby('TARGET')['AMT_INCOME_TOTAL'].mean().values,
    palette='pastel'
)
plt.title("Mean Income by Default Status")
plt.xlabel("TARGET (0 = Non-Default, 1 = Default)")
plt.ylabel("Mean Income")
plt.show()

df_eda[['TARGET','EXT_SOURCE_2','EXT_SOURCE_3','AGE','EMPLOY_YEARS']].corr()

plt.figure(figsize=(6,4))
sns.heatmap(
    df_eda[['TARGET','EXT_SOURCE_2','EXT_SOURCE_3','AGE','EMPLOY_YEARS']].corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation with Default (TARGET)")
plt.show()

from sklearn.model_selection import train_test_split

X = df_model.drop(columns=['TARGET'])
y = df_model['TARGET']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train.shape, X_test.shape

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train_scaled, y_train)

y_pred_log = log_model.predict(X_test_scaled)
y_proba_log = log_model.predict_proba(X_test_scaled)[:,1]

print("Logistic Regression Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Precision:", precision_score(y_test, y_pred_log))
print("Recall:", recall_score(y_test, y_pred_log))
print("F1 Score:", f1_score(y_test, y_pred_log))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_log))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))

X_train_knn = X_train_scaled[:20000]
y_train_knn = y_train[:20000]

X_test_knn = X_test_scaled[:5000]
y_test_knn = y_test[:5000]

print("KNN training size:", X_train_knn.shape)
print("KNN testing size:", X_test_knn.shape)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_knn, y_train_knn)

y_pred_knn = knn.predict(X_test_knn)
y_proba_knn = knn.predict_proba(X_test_knn)[:,1]

print("KNN Metrics:")
print("Accuracy:", accuracy_score(y_test_knn, y_pred_knn))
print("Precision:", precision_score(y_test_knn, y_pred_knn))
print("Recall:", recall_score(y_test_knn, y_pred_knn))
print("F1 Score:", f1_score(y_test_knn, y_pred_knn))
print("ROC-AUC:", roc_auc_score(y_test_knn, y_proba_knn))

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=6, random_state=42)
tree.fit(X_train, y_train)

y_pred_tree = tree.predict(X_test)
y_proba_tree = tree.predict_proba(X_test)[:,1]

print("Decision Tree Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print("Precision:", precision_score(y_test, y_pred_tree))
print("Recall:", recall_score(y_test, y_pred_tree))
print("F1 Score:", f1_score(y_test, y_pred_tree))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_tree))



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


acc_log  = accuracy_score(y_test, y_pred_log)
prec_log = precision_score(y_test, y_pred_log)
rec_log  = recall_score(y_test, y_pred_log)
f1_log   = f1_score(y_test, y_pred_log)
auc_log  = roc_auc_score(y_test, y_proba_log)


acc_knn  = accuracy_score(y_test_knn, y_pred_knn)
prec_knn = precision_score(y_test_knn, y_pred_knn)
rec_knn  = recall_score(y_test_knn, y_pred_knn)
f1_knn   = f1_score(y_test_knn, y_pred_knn)
auc_knn  = roc_auc_score(y_test_knn, y_proba_knn)


acc_tree  = accuracy_score(y_test, y_pred_tree)
prec_tree = precision_score(y_test, y_pred_tree)
rec_tree  = recall_score(y_test, y_pred_tree)
f1_tree   = f1_score(y_test, y_pred_tree)
auc_tree  = roc_auc_score(y_test, y_proba_tree)

import pandas as pd
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN (subset)', 'Decision Tree'],
    'Accuracy': [acc_log, acc_knn, acc_tree],
    'Precision': [prec_log, prec_knn, prec_tree],
    'Recall': [rec_log, rec_knn, rec_tree],
    'F1 Score': [f1_log, f1_knn, f1_tree],
    'ROC-AUC': [auc_log, auc_knn, auc_tree]
})
results

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

# Logistic
fpr_log, tpr_log, _ = roc_curve(y_test, y_proba_log)
plt.plot(fpr_log, tpr_log, label=f"Logistic Regression (AUC={auc(fpr_log, tpr_log):.3f})")

# Decision Tree
fpr_tree, tpr_tree, _ = roc_curve(y_test, y_proba_tree)
plt.plot(fpr_tree, tpr_tree, label=f"Decision Tree (AUC={auc(fpr_tree, tpr_tree):.3f})")

# KNN (subset)
fpr_knn, tpr_knn, _ = roc_curve(y_test_knn, y_proba_knn)
plt.plot(fpr_knn, tpr_knn, label=f"KNN Subset (AUC={auc(fpr_knn, tpr_knn):.3f})", linestyle='--')

plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()

import seaborn as sns
from sklearn.metrics import confusion_matrix

cm_log = confusion_matrix(y_test, y_pred_log)

plt.figure(figsize=(5,4))
sns.heatmap(cm_log, annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()