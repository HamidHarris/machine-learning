import pandas as pd
import numpy as np
print(pd.__version__)
pd.set_option('display.max_columns', None)

df = pd.read_csv('./car_fuel_efficiency.csv')
print(df.shape)

print(df.head())
print("\n")

print(f"Number of types of fuels = {df['fuel_type'].nunique()}")

print(df.isna().sum())

print(df.groupby('origin').max())

print(f"Old median: {df.horsepower.median()}")

most_frequent_horsepower = df.horsepower.mode()[0]
df['horsepower'] = df['horsepower'].fillna(most_frequent_horsepower)

print(f"New median: {df.horsepower.median()}")



X = df.loc[df['origin'] == 'Asia', ['vehicle_weight', 'model_year']].head(7).values
print(X)

XTX = X.T.dot(X)
print(XTX)

XTX_inv = np.linalg.inv(XTX)

y = [1100, 1300, 800, 900, 1000, 1100, 1200]

w = XTX_inv.dot(X.T).dot(y)
print(w)

print(w.sum())