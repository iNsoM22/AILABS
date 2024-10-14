import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

pd.set_option('display.max_columns', None)
df = pd.read_csv('weatherAUS.csv')

df.info()
df = df.drop('Date', axis=1)

df.columns = df.columns.str.lower()
df['raintoday'] = df['raintoday'].fillna('No')
df['raintomorrow'] = df['raintomorrow'].fillna('No')


def add_prefix(data: pd.DataFrame, column: str, prefix: str):
    return data[column].apply(lambda x: prefix + str(x))


df['winddir9am'] = add_prefix(df, 'winddir9am', '9_')
df['winddir3pm'] = add_prefix(df, 'winddir3pm', '3_')


def filling(data: pd.DataFrame, target):
    return data[target].fillna(df[target].mean())


df.select_dtypes(np.number).isna().sum() > 1

for col in df.select_dtypes(np.number).columns:
    df[col] = filling(df, col)


def one_hot_encoding(data: pd.DataFrame, features: str | list):
    for col in features:
        dummy = pd.get_dummies(data[col], dtype=int)
        data = pd.concat([data, dummy], axis=1)
        data.drop(col, axis=1, inplace=True)
    return data


features_to_encode = ['location', 'winddir9am', 'winddir3pm', 'windgustdir']
df = one_hot_encoding(df, features_to_encode)

label = LabelEncoder()
df['raintoday'] = label.fit_transform(df['raintoday'])
df['raintomorrow'] = label.fit_transform(df['raintomorrow'])

X = df.drop('raintomorrow', axis=1)
y = df['raintomorrow']

scaler = RobustScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    random_state=123,
                                                    shuffle=True,
                                                    stratify=y,
                                                    test_size=0.2)


X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
