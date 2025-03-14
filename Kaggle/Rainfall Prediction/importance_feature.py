import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import zscore
from imblearn.over_sampling import SMOTE
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# =================================================================

train_path = "D:\Kaggle\Binary Prediction with a Rainfall Dataset/train.csv"
rain_path = "D:\Kaggle\Binary Prediction with a Rainfall Dataset/Rainfall.csv"

df = pd.read_csv(train_path).drop('id', axis=1)
rf = pd.read_csv(rain_path).dropna(axis=0)


def read_data(path, nhead=10):
    df = pd.read_csv(path)
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    print(df.info())
    print(df.describe())
    print(df['rainfall'].value_counts())
    print(df['rainfall'].value_counts())
    print(df.isna().sum())
    print(df.head(nhead))
    print(df.columns)

    return df


df = read_data(train_path)
rf = read_data(rain_path)
rf['day'] = list(range(1, 367))

# ================================= Merging rf and df ===================================

data = pd.concat([rf, df], axis=0)
data['cloud'].head(10)
data['cloud'].describe()

len(data[(data['cloud'] < 68) & (data['rainfall'] == 1)])
len(data[(data['cloud'] < 68) & (data['rainfall'] == 0)])

len(data[(data['cloud'] > 68) & (data['cloud'] < 83) & (data['rainfall'] == 1)])
len(data[(data['cloud'] > 68) & (data['cloud'] < 83) & (data['rainfall'] == 0)])

len(data[(data['cloud'] > 83) & (data['cloud'] < 88) & (data['rainfall'] == 1)])
len(data[(data['cloud'] > 83) & (data['cloud'] < 88) & (data['rainfall'] == 0)])

len(data[(data['cloud'] > 88) & (data['rainfall'] == 1)])
len(data[(data['cloud'] > 88) & (data['rainfall'] == 0)])

data['sunshine'].describe()

plt.scatter(data['sunshine'], data['cloud'])
plt.scatter(data['pressure'], data['temparature'], label=data['rainfall'])
sns.scatterplot(x='pressure', y='temparature', data=df, hue='rainfall')
plt.xlabel('Pressure')
plt.ylabel('Temperature')
plt.legend()
plt.show()

# The data is stored in a dictionary-like structure
# Print the keys to see what variables are stored in the file
print(data.keys())

# Access specific variables
# For example, if there's a variable named 'interference_data'
if 'interference_data' in data:
    interference_data = data['interference_data']
    print(interference_data)
else:
    print("The variable 'interference_data' does not exist in the file.")

df.corr()
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

significant_features = [
    'cloud',  # High correlation, importance, and mutual information
    'sunshine',  # High negative correlation and mutual information
    'humidity',  # Moderate correlation and high mutual information
    'dewpoint',  # Moderate correlation and importance
    'windspeed',  # Moderate correlation and importance
    'rainfall'  # Target variable
]

# Drop less significant features
cleaned_data = data[significant_features]

import seaborn as sns
import matplotlib.pyplot as plt

# Correlation matrix
corr = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

sns.pairplot(data, x_vars=['temparature', 'humidity', 'pressure'], y_vars=['rainfall'])
plt.show()

from scipy.stats import pearsonr

numerical_features = data.select_dtypes(include=['int', 'float']).columns.tolist()
categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()

for col in numerical_features:
    corr, p_value = pearsonr(data[col], data['rainfall'])
    print(f"{col}: Correlation = {corr:.2f}, p-value = {p_value:.4f}")

from scipy.stats import f_oneway

for col in categorical_features:
    groups = [data[data[col] == category]['rainfall'] for category in data[col].unique()]
    f_stat, p_value = f_oneway(*groups)
    print(f"{col}: F-statistic = {f_stat:.2f}, p-value = {p_value:.4f}")

from sklearn.ensemble import RandomForestRegressor

# Prepare data
X = data.drop(columns=['rainfall'])
y = data['rainfall']

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Feature importance
importance = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)

from sklearn.linear_model import Lasso

# Train model
model = Lasso(alpha=0.01)
model.fit(X, y)

# Coefficients
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
print(coef_df)

from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # Retain 95% of variance
X_pca = pca.fit_transform(X)
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

model = LinearRegression()
rfe = RFE(model, n_features_to_select=10)  # Select top 10 features
X_rfe = rfe.fit_transform(X, y)
print(f"Selected features: {X.columns[rfe.support_]}")

from sklearn.feature_selection import mutual_info_regression

mi = mutual_info_regression(X, y)
mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mi})
mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)
print(mi_df)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.show()

