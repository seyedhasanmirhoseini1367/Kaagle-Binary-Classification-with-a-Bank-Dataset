import pandas as pd

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
rf['day'] = list(range(1, 366))

# ================================= Merging rf and df ===================================

data = pd.concat([rf, df], axis=0)

data.to_csv('D:\Kaggle\Binary Prediction with a Rainfall Dataset/train.csv', index=False)
