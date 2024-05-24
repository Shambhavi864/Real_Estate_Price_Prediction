import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20, 20)

#DATA CLEANING
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

#FEATURE ENGINNERING & DIMENTIONALITY REDUCTION TECHNIQUES
df5=df4.copy()

# new feature
df5['price_per_sqrft'] = df5['price']*100000/df5['total_sqft']
# len(df5.location.unique())
df5.location=df5.location.apply(lambda x: x.strip())

location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)

location_stats_less_than_10 =location_stats[location_stats<=10]
location_stats_less_than_10

df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
# Eliminating those locations which has location stats less than 10
len(df5.location.unique())


#OUTLIER REMOVAL
df6 = df5[~(df5.total_sqft/df5.bhk<300)]

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean (subdf.price_per_sqrft)
        st = np.std (subdf.price_per_sqrft)
        reduced_df = subdf[(subdf.price_per_sqrft>(m-st)) & (subdf.price_per_sqrft<=(m+st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out
    
df7 = remove_pps_outliers(df6)

#Drawing scatter graph
def plot_scatter_chart (df, location):
    bhk2= df[(df.location ==location) & (df.bhk==2)] 
    bhk3= df[(df.location==location) & (df.bhk==3)] 
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter (bhk2.total_sqft, bhk2.price_per_sqrft, color='blue', label='2 BHK', s=50)
    plt.scatter (bhk3.total_sqft, bhk3.price_per_sqrft,marker='+', color='green', label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price")
    plt.title(location)
    plt.legend()

def remove_bhk_outliers (df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'): 
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'): 
            bhk_stats[bhk] = {
                'mean': np.mean (bhk_df.price_per_sqrft),
                'std': np.std(bhk_df.price_per_sqrft), 
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats=bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df [bhk_df.price_per_sqrft < (stats['mean'])]. index.values) 
    return df.drop(exclude_indices, axis='index')
df8 = remove_bhk_outliers (df7)
df8.shape

matplotlib.rcParams["figure.figsize"] = (20, 10)

df9=df8[df8.bath<df8.bhk+2]

df10= df9.drop(['size', 'price_per_sqrft'], axis = 'columns')
