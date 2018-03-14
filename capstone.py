import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True)
from pandas.plotting import scatter_matrix
from scipy import stats
from sklearn.linear_model import LogisticRegression

#load data
df = pd.read_csv('data/09-13.csv')

def clean_data(df):
    #subset by hourly and meta data
    mycols = [
           'STATION_NAME', 'ELEVATION', 'DATE', 'REPORTTPYE', 'HOURLYSKYCONDITIONS', 'HOURLYVISIBILITY','HOURLYPRSENTWEATHERTYPE', 'HOURLYDRYBULBTEMPF', 'HOURLYWETBULBTEMPF', 'HOURLYDewPointTempF', 'HOURLYRelativeHumidity', 'HOURLYWindSpeed', 'HOURLYWindDirection', 'HOURLYWindGustSpeed', 'HOURLYStationPressure', 'HOURLYPressureTendency', 'HOURLYPressureChange','HOURLYSeaLevelPressure', 'HOURLYPrecip'
    ]
    df = df[mycols]
    renamed_cols = {
        'STATION_NAME':'station_name', 'ELEVATION':'elevation', 'DATE':'date', 'REPORTTPYE':'report_type', 'HOURLYSKYCONDITIONS':'sky_conditions', 'HOURLYVISIBILITY': 'visibility','HOURLYPRSENTWEATHERTYPE': 'weather_type', 'HOURLYDRYBULBTEMPF': 'dry_bulb_tmpF','HOURLYWETBULBTEMPF':'wet_bulb_tmpF', 'HOURLYDewPointTempF':'dew_point_tmpF', 'HOURLYRelativeHumidity':'humidity','HOURLYWindSpeed':'wind_speed', 'HOURLYWindDirection':'wind_direction', 'HOURLYWindGustSpeed':'gust_speed', 'HOURLYStationPressure':'pressure', 'HOURLYPressureTendency':'pressure_tedency', 'HOURLYPressureChange':'pressure_change','HOURLYSeaLevelPressure':'sea_lvl_pressure', 'HOURLYPrecip':'precipitation'
    }
    df.rename(index=str, columns=renamed_cols, inplace=True)
    # df.rename(str.lower, axis='columns', inplace=True)

    #clean precip
    df.loc[df['precipitation'] == "T", :] = 0
    df.loc[:, 'precipitation'] = df['precipitation'].map(lambda x: x if isinstance(x,(int,float)) else x.lstrip('*').rstrip('Vs'))
    df['precipitation'] = pd.to_numeric(df['precipitation'])

    #clean visibility
    df.loc[:,'visibility'] = df.loc[:,'visibility'].map(lambda x: x if isinstance(x, (int,float)) else x.lstrip('*').rstrip('Vs'))
    df['visibility'] = pd.to_numeric(df['visibility'])

    #clean dry bulb tmp
    df.loc[:,'dry_bulb_tmpF'] = df['dry_bulb_tmpF'].map(lambda x: x if isinstance(x, (int,float)) else x.lstrip('*').rstrip('Vs'))
    df['dry_bulb_tmpF'] = pd.to_numeric(df['dry_bulb_tmpF'])

    #clean dew point tmp
    df.loc[:,'dew_point_tmpF'] = df['dew_point_tmpF'].map(lambda x: x if isinstance(x, (int,float)) else x.lstrip('*').rstrip('Vs'))
    df['dew_point_tmpF'] = pd.to_numeric(df['dew_point_tmpF'])

    #clean wind direction
    df.loc[:,'wind_direction'] = df['wind_direction'].map(lambda x: x if isinstance(x, (int,float)) else x.lstrip('*').rstrip('VRBs'))
    df.loc[df['wind_direction'] == '', 'wind_direction'] = 0
    df['wind_direction'] = pd.to_numeric(df['wind_direction'])

    #clean wind direction
    df.loc[:,'wind_speed'] = df['wind_speed'].map(lambda x: x if isinstance(x, (int,float)) else x.lstrip('*').rstrip('VRBs'))
    df['wind_speed'] = pd.to_numeric(df['wind_speed'])

    #clean pressure`
    df.loc[:,'pressure'] = df['pressure'].map(lambda x: x if isinstance(x, (int,float)) else x.lstrip('*').rstrip('s'))
    df['pressure'] = pd.to_numeric(df['pressure'])

    #clean pressure change
    df.loc[:,'pressure_change'] = df['pressure_change'].map(lambda x: x if isinstance(x, (int,float)) else x.lstrip('+*').rstrip('s'))
    df['pressure_change'] = pd.to_numeric(df['pressure_change'])

    #clean sea lvl pressure
    df.loc[:,'sea_lvl_pressure'] = df['sea_lvl_pressure'].map(lambda x: x if isinstance(x, (int,float)) else x.lstrip('*').rstrip('s'))
    df['sea_lvl_pressure'] = pd.to_numeric(df['sea_lvl_pressure'])

    return df

def clean_num_var(col):
    df.loc[:,col] = df[col].map(lambda x: x if isinstance(x, (int,float)) else x.lstrip('+*').rstrip('VRBs'))
    df[col] = pd.to_numeric(df[col])


df = clean_data(df)

def create_lag_var(df, col, lag):
    n = df.shape[0]
    newvar = [None]*lag + [df[col][i-lag] for i in range(lag, n)]
    col_name = f"{col}{n}"
    df[col_name] = newvar

# Create will rain target variable: A lagged by one hour will it rain dichotmous var
df["will_rain"] = [df['precipitation'][i+1] > 0 for i in range(0, df.shape[0]-1)] + [None]
df.drop(df.tail(1).index,inplace=True)


def get_null_perc(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data

get_null_perc(df)

df.drop(['weather_type'], axis=1, inplace=True)


df.corr()
# scattermatrix = scatter_matrix(df.select_dtypes(include=['float']), alpha=0.2,figsize=(8,8), diagonal="hist")


# Correlation heat map
# correlation_map = np.corrcoef(df.select_dtypes(include=['float']).values.T)
# sns.set(font_scale=1.0)
# heatmap = sns.heatmap(correlation_map, cbar=True, annot=True, square=True, fmt='.2f', yticklabels=columns.values, xticklabels=columns.values)
# plt.show()

x_cols = ['visibility', 'dry_bulb_tmpF', 'wet_bulb_tmpF', 'dew_point_tmpF', 'humidity','wind_speed', 'wind_direction', 'gust_speed', 'pressure', 'pressure_tedency', 'pressure_change','sea_lvl_pressure', 'precipitation']

for x in x_cols:
    r,p = stats.pointbiserialr(df['will_rain'], df[x])
    print(f'var = {x}, r = {r}, p = {p}')

# SPLIT DATA HERE

## Run log reg on all the data
classifier = LogisticRegression()
y_pred = classifier.fit(X,y).predict(X)
y_true = y

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
print(cnf_matrix)

#Normalized Confusion Matrix
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111)
ax.grid(False)
class_names = ['No Rain',"Rain"]
plot_confusion_matrix(cnf_matrix, ax, classes=class_names,normalize=True,
                      title='Normalized Confusion matrix')
#
# loss  = make_scorer(rmsle, greater_is_better=False)
# #How to manage many models
# pipelines = []
# pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))
# pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
# pipelines.append(('ScaledRidge', Pipeline([('Scaler', StandardScaler()),('EN', Ridge())])))
#
#
# results = []
# names = []
# for name, model in pipelines:
#     kfold = KFold(n_splits=10, random_state=21)
#     cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=loss)
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#     print(msg)
#
