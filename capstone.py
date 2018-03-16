import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True)
from pandas.plotting import scatter_matrix
from scipy import stats

from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
from sklearn.base import clone

from basis_expansions import NaturalCubicSpline
from src.dftransformers import (ColumnSelector, Identity,
                            FeatureUnion, MapFeature,
                            Intercept)
from src.regression_helpers import (plot_univariate_smooth,
                                bootstrap_train,
                                display_coef,
                                plot_bootstrap_coefs,
                                plot_partial_depenence,
                                plot_partial_dependences,
                                predicteds_vs_actuals)
import statsmodels.api as sm
import itertools as itertools

def clean_data(df):

    #subset by hourly and meta data
    mycols = [
           'STATION_NAME', 'DATE', 'REPORTTPYE', 'HOURLYSKYCONDITIONS', 'HOURLYVISIBILITY','HOURLYPRSENTWEATHERTYPE', 'HOURLYDRYBULBTEMPF', 'HOURLYWETBULBTEMPF', 'HOURLYDewPointTempF', 'HOURLYRelativeHumidity', 'HOURLYWindSpeed', 'HOURLYWindDirection', 'HOURLYWindGustSpeed', 'HOURLYStationPressure', 'HOURLYPressureTendency', 'HOURLYPressureChange','HOURLYSeaLevelPressure', 'HOURLYPrecip'
    ]
    df = df[mycols]

    #column names that are more to my likeing
    renamed_cols = {
        'STATION_NAME':'station_name', 'DATE':'datetime', 'REPORTTPYE':'report_type', 'HOURLYSKYCONDITIONS':'sky_conditions', 'HOURLYVISIBILITY': 'visibility','HOURLYPRSENTWEATHERTYPE': 'weather_type', 'HOURLYDRYBULBTEMPF': 'dry_bulb_tmpF','HOURLYWETBULBTEMPF':'wet_bulb_tmpF', 'HOURLYDewPointTempF':'dew_point_tmpF', 'HOURLYRelativeHumidity':'humidity','HOURLYWindSpeed':'wind_speed', 'HOURLYWindDirection':'wind_direction', 'HOURLYWindGustSpeed':'gust_speed', 'HOURLYStationPressure':'pressure', 'HOURLYPressureTendency':'pressure_tedency', 'HOURLYPressureChange':'pressure_change','HOURLYSeaLevelPressure':'sea_lvl_pressure', 'HOURLYPrecip':'precipitation'
    }
    df.rename(index=str, columns=renamed_cols, inplace=True)

    #create date var
    df['datetime'] = pd.to_datetime(df['datetime'])

    #clean precip
    df.loc[df['precipitation'] == "T", :] = 0
    df.loc[:, 'precipitation'] = df['precipitation'].map(lambda x: x if isinstance(x,(int,float)) else x.lstrip('*').rstrip('Vs'))
    df['precipitation'] = pd.to_numeric(df['precipitation'])

    #clean visibility
    df.loc[:,'visibility'] = df['visibility'].map(lambda x: x if isinstance(x, (int,float)) else x.lstrip('*').rstrip('Vs'))
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
    df['northward'] = np.sin(df['wind_direction'])
    df['eastward'] = np.cos(df['wind_direction'])
    df.drop(['wind_direction'], axis=1, inplace=True)

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

    # recode cloud condtions into ordinal
    def clean_cloud_conditions(string):
        if string == np.nan:
            return string
        x = 0
        if isinstance(string, str):
            if "FEW" in string:
                x +=1
            if "SCT" in string:
                x +=2
            if "BKW" in string:
                x +=3
            if "OVC" in string:
                x +=4
        return x

    df.loc[:,'sky_conditions'] = df['sky_conditions'].map(clean_cloud_conditions)


    return df

def agg_data(df):
    agg_funcs = ['mean', 'max', 'min']
    result = df.groupby(df['datetime'].dt.date).agg(agg_funcs)
    result.columns = ['_'.join(col).strip() for col in result.columns.values]
    return result

def clean_num_var(col):
    df.loc[:,col] = df[col].map(lambda x: x if isinstance(x, (int,float)) else x.lstrip('+*').rstrip('VRBs'))
    df[col] = pd.to_numeric(df[col])

def create_lag_var(df, col, lag):
    n = df.shape[0]
    newvar = [None]*lag + [df[col][i-lag] for i in range(lag, n)]
    col_name = f"{col}{n}"
    df[col_name] = newvar

def get_null_perc(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data

def corr_heat(df):
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(12, 12))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},xticklabels=corr.index, yticklabels=corr.columns)
    plt.xticks(rotation=60, ha="right")
    plt.yticks(rotation=0)
    ax.set_title("Correlation Heat Map")

def create_scatter_matrix(df):
    myvars = np.absolute(df.corr()['precipitation mean']).sort_values()[-8:-1]
    axs = scatter_matrix(df[myvars.index], alpha=0.2,figsize=(8,8), diagonal="hist")
    n = len(myvars.index)
    for x in range(n):
        for y in range(n):
            # to get the axis of subplots
            ax = axs[x, y]
            # to make x axis name vertical
            ax.xaxis.label.set_rotation(75)
            # to make y axis name horizontal
            ax.yaxis.label.set_rotation(0)
            # to make sure y axis names are outside the plot area
            ax.yaxis.labelpad = 50

def simple_spline_specification(name, knots):
    select_name = "{}_select".format(name)
    spline_name = "{}_spline".format(name)
    return Pipeline([
        (select_name, ColumnSelector(name=name)),
        (spline_name, NaturalCubicSpline(knots=knots))
    ])

def plot_one_univariate(ax, var_name, mask=None):
    if mask is None:
        plot_univariate_smooth(
            ax,
            df[var_name].values.reshape(-1, 1),
            df['Balance_log'],
            bootstrap=200)
    else:
        plot_univariate_smooth(
            ax,
            df[var_name].values.reshape(-1, 1),
            df['Balance_log'],
            mask=mask,
            bootstrap=200)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def cv(X, y, base_estimator, n_folds, random_seed=154):
    """Estimate the in and out-of-sample error of a model using cross validation.

    Parameters
    ----------

    X: np.array
      Matrix of predictors.

    y: np.array
      Target array.

    base_estimator: sklearn model object.
      The estimator to fit.  Must have fit and predict methods.

    n_folds: int
      The number of folds in the cross validation.

    random_seed: int
      A seed for the random number generator, for repeatability.

    Returns
    -------

    train_cv_errors, test_cv_errors: tuple of arrays
      The training and testing errors for each fold of cross validation.
    """
    kf = KFold(n_splits=n_folds, random_state=random_seed)
    test_cv_errors, train_cv_errors = np.empty(n_folds), np.empty(n_folds)
    accuracies = []
    precisions = []
    recalls = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for idx, (train, test) in enumerate(kf.split(X)):
        # Split into train and test
        X_cv_train, y_cv_train = X[train], y[train]
        X_cv_test, y_cv_test = X[test], y[test]
        # Standardize data.
        # standardizer = StandardScaler()
        # standardizer.fit(X_cv_train, y_cv_train)
        # X_cv_train_std = standardizer.transform(X_cv_train)
        # X_cv_test_std = standardizer.transform(X_cv_test)
        # Fit estimator
        estimator = clone(base_estimator)
        estimator.fit(X_cv_train, y_cv_train)
        # Measure performance
        y_hat_train = estimator.predict(X_cv_train)
        y_predict = estimator.predict(X_cv_test)
        # Calclate the error metrics
        probas_ = estimator.predict_proba(X_cv_test)[:,1]
        accuracies.append(metrics.accuracy_score(y_cv_test, y_predict))
        precisions.append(metrics.precision_score(y_cv_test, y_predict))
        recalls.append(metrics.recall_score(y_cv_test, y_predict))

        fpr, tpr, thresholds = metrics.roc_curve(y_cv_test, probas_)
        aucs.append(metrics.auc(fpr, tpr))
    return {"auc": np.mean(aucs), "accuracy":np.mean(accuracies), "precision":np.mean(precisions), "recall": np.mean(recalls)}

#load data
df = pd.read_csv('data/data.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df.drop(['Unnamed: 0'], axis=1, inplace=True)
df['northward'] = np.sin(df['wind_direction'])
df['eastward'] = np.cos(df['wind_direction'])
df.drop(['wind_direction'], axis=1, inplace=True)


#clean data
# df = clean_data(df)
daily = agg_data(df)
daily.drop(['gust_speed_mean', 'gust_speed_min', 'gust_speed_max'], axis=1, inplace=True)
daily = daily.dropna()

#split data
x_cols = ["precipitation" not in i for i in daily.columns]
X = daily.loc[:,x_cols]
y = daily['precipitation_mean'] > 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

# EDA STUFF
r_list = []
p_list = []
for x in X.columns:
    r, p = stats.pointbiserialr(y_train, X_train[x])
    r_list.append(r)
    p_list.append(p)
pb_df = pd.DataFrame({'Var':X.columns, 'R':r_list, 'p':p_list})
pb_df = pb_df[["Var", "R", "p"]]
pb_df.reindex(pb_df['R'].abs().sort_values(ascending=False, inplace=False).index).head(8).round(3).to_html()
df.corr()
# corr_heat(X_train)
get_null_perc(X_train)
df.describe()

# plot histogram
fig, ax = plt.subplots()
_ = ax.hist(y_train, bins=50)
y_log = np.log(y_train + .00000000000000001)
fig, ax = plt.subplots()
_ = ax.hist(y_log, bins=50)

# # univariate plots
# mycols_list = [x_cols[i:i + 3] for i in range(0, len(x_cols), 3)]
# for i in mycols_list:
#     plot_univariate(y, i)


# Run Logistic Lasso
standardizer = StandardScaler()
standardizer.fit(X_train)
X_train_std = standardizer.transform(X_train)
X_test_std = standardizer.transform(X_test)
# Fit estimator
estimator = LogisticRegression(penalty="l1")
estimator.fit(X_train_std, y_train)

#build my pipeline
pipelines = []
pipelines.append(('Log Regression(Ridge)', Pipeline([('Scaler', StandardScaler()),('Ridge',LogisticRegression(penalty="l2"))])))
pipelines.append(('Log Regression(Lasso)', Pipeline([('Scaler', StandardScaler()),('Lasso', LogisticRegression(penalty="l1"))])))
pipelines.append(('Log Regression', Pipeline([('Scaler', StandardScaler()),('Log', LogisticRegression(penalty="l2", C=99999999))])))

#
# results = []
# names = []
results = dict()
for name, model in pipelines:
    results[name] = cv(X_train.values, y_train.values, model, 8, 10)

standardizer = StandardScaler()
standardizer.fit(X_train)
X_train_std = standardizer.transform(X_train)
X_test_std = standardizer.transform(X_test)
# Fit estimator
estimator = LogisticRegression()
estimator.fit(X_train_std, y_train)

coef_dict = {}
for coef, feat in zip(estimator.coef_[0],X_train.columns):
    coef_dict[feat] = coef
for k, v in sorted(coef_dict.items(), key=lambda x:abs(x[1])):
    print(f"{k}: {v:.2f}")
mycoefs = pd.DataFrame({'Beta':list(coef_dict.values()), index= coef_dict.keys())
mycoefs.reindex(mycoefs['Beta'].abs().sort_values(ascending=
False, inplace=False).index)


leave_out = [k for k,v in coef_dict.items() if v == 0]
paired_X = X_train.loc[:,~np.in1d(X_train.columns, leave_out)]
paired_results = cv(paired_X.values, y_train.values, model, 5, 10)
#
# train
date = pd.Series(X_train.index.values)
month = pd.to_datetime(date).dt.month
month_results = dict()
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul","Aug", "Sep", "Oct", "Nov", "Dec

#test
date_test = pd.Series(X_test.index.values)
month_test = pd.to_datetime(date).dt.month


month_models = []
model = Pipeline([('Scaler', StandardScaler()),('Log', LogisticRegression(penalty="l2"))])
test_month_aucs = {}
for i,name in enumerate(months):
    mask = month == (i+1)
    test_mask = month_test == (i+1)
    month_X = X_train.reset_index(drop=True)[mask]
    month_y = y_train.reset_index(drop=True)[mask]
    estimator = clone(model)
    estimator.fit(month_X, month_y)

    # Calclate the error metrics
    probas_ = estimator.predict_proba(X_test.reset_index(drop=True)[test_mask])[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test.reset_index(drop=True)[test_mask], probas_)
    test_month_aucs[name] = metrics.auc(fpr, tpr)
    month_results[name] = cv(month_X.values, month_y.values, model, 8,3)['auc']

model = Pipeline([('Scaler', StandardScaler()),('Log', LogisticRegression())])
estimator = clone(model)
estimator.fit(X_train, y_train)
probas_ = estimator.predict_proba(X_test)[:,1]
fpr, tpr, thresh = metrics.roc_curve(y_test, probas_)
test_auc = metrics.auc(fpr, tpr)

tampa = pd.read_csv('data/tampa.csv')
tampa.drop(['Unnamed: 0'], axis=1, inplace=True)
tampa['northward'] = np.sin(tampa['wind_direction'])
tampa['eastward'] = np.cos(tampa['wind_direction'])
tampa.drop(['wind_direction'], axis=1, inplace=True)

#clean data
# df = clean_data(df)
daily_tampa = agg_data(tampa)
daily_tampa.drop(['gust_speed_mean', 'gust_speed_min', 'gust_speed_max'], axis=1, inplace=True)
daily_tampa = daily_tampa.dropna()

#split data
x_cols = ["precipitation" not in i for i in daily.columns]
X_tampa = daily_tampa.loc[:,x_cols]
y_tampa = daily_tampa['precipitation_mean'] > 0
probas_ = estimator.predict_proba(X_tampa)[:,1]
fpr, tpr, thresh = metrics.roc_curve(y_tampa, probas_)
tampa_auc = metrics.auc(fpr, tpr)

#test
date_tampa= pd.Series(X_tampa.index.values)
month_tampa = pd.to_datetime(date_tampa).dt.month
tampa_month_aucs = {}
for i,name in enumerate(months):
    mask = month == (i+1)
    tampa_mask = month_tampa == (i+1)
    month_X = X_train.reset_index(drop=True)[mask]
    month_y = y_train.reset_index(drop=True)[mask]
    estimator = clone(model)
    estimator.fit(month_X, month_y)

    # Calclate the error metrics
    probas_ = estimator.predict_proba(X_tampa.reset_index(drop=True)[tampa_mask])[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(y_tampa.reset_index(drop=True)[tampa_mask], probas_)
    tampa_month_aucs[name] = metrics.auc(fpr, tpr)
    month_results[name] = cv(month_X.values, month_y.values, model, 8,3)['auc']
final_results = pd.DataFrame([month_results, test_month_aucs], index=["Train", "Test"])
final_results['Overall'] = [94.50,test_auc]
final_results.loc['tampa'] = tampa_month_aucs

# # Create will rain target variable: A lagged by one hour will it rain dichotmous var
# df["will_rain"] = [df['precipitation'][i+1] > 0 for i in range(0, df.shape[0]-1)] + [None]
# df.drop(df.tail(1).index,inplace=True)





# #How to manage many models
# pipelines = []
# pipelines.append(('Log Regression(Ridge)', Pipeline([('Scaler', StandardScaler()),('Ridge',LogisticRegression(penalty="l2"))])))
# pipelines.append(('Log Regression(Lasso)', Pipeline([('Scaler', StandardScaler()),('Lasso', LogisticRegression(penalty="l1"))])))
# pipelines.append(('Log Regression', Pipeline([('Scaler', StandardScaler()),('Log', LogisticRegression(penalty="l2", C=99999999))])))
#
# #
# # results = []
# # names = []
# results = dict()
# for name, model in pipelines:
#     results[name] = cv(X_train.values, y_train.values, model, 5, 10)
    # kfold = KFold(n_splits=10, random_state=21)
    # cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=loss)
    # results.append(cv_results)
    # names.append(name)
    # msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    # print(msg)
#
