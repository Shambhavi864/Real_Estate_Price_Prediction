import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20, 20)

df1 = pd.read_csv("Bengaluru_House_Data.csv")
df1.head()

df1.groupby('area_type')['area_type'].agg('count')

df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis = 'columns')
df2.head()

df3 = df2.dropna()
df3.isnull().sum()

df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))

df3 = df2.dropna()
df3.isnull().sum()
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))

def convert_sqft_to_nm(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    # If a no is like 1220
    try:
        return float(x)
    except:
        return None

df4=df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_nm)
df4.head(50)