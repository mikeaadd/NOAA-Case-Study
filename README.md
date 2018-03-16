# NOAA-Case-Study: Will It Rain?

1. Does Temperature, sunshine, humidity, wind direction and air pressure predict precipitation in the Denver Airport Weather Station

2. If Temperature, sunshine, humidity, wind direction and air pressure does predict precipitation which predictor is the most significant

# The Data

- hourly weather data from the Denver Airport Weather Station from 1989 to 2018

- 35800 rows and 19 columns

- station_name', 'datetime', 'report_type', 'sky_conditions', 'visibility', 'weather_type', 'dry_bulb_tmpF', 'wet_bulb_tmpF',
'dew_point_tmpF', 'humidity', 'wind_speed', 'gust_speed', 'pressure', 'pressure_tedency', 'pressure_change', 'sea_lvl_pressure','precipitation', 'northward', 'eastward'

- missing data in primarily gust_speed and weather type

<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th></th>\n      <th>Total</th>\n      <th>Percent</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>gust_speed</th>\n      <td>310749</td>\n      <td>0.868</td>\n    </tr>\n    <tr>\n      <th>weather_type</th>\n      <td>301449</td>\n      <td>0.842</td>\n    </tr>\n    <tr>\n      <th>pressure_change</th>\n      <td>238747</td>\n      <td>0.667</td>\n    </tr>\n    <tr>\n      <th>pressure_tedency</th>\n      <td>204502</td>\n      <td>0.571</td>\n    </tr>\n    <tr>\n      <th>precipitation</th>\n      <td>126642</td>\n      <td>0.354</td>\n    </tr>\n  </tbody>\n</table>

- all numeric except weather type and sky conditions

### Feature Engineering

- aggregated all data into daily data

- created min, mean, max vars for all numeric

- Wind Direction was a directional variable (in degrees)
    * transformed variable into Eastward and Northward variables by taking the cos and sin

- Transformed Sky conditions into ordinal
    1. Clear Sky: 0
    2. Few: 1
    3. Scattered: 2
    4. Broken Clouds: 3
    5. Overcast: 4

# EDA

- Before any EDA the data was split with a 20% holdout

- Correlation Heatmap

![alt text](https://github.com/mikeaadd/NOAA-Case-Study/img/hist_y.png "Correlation Heat Map")

- Histogram of target Variable, precipitation

![alt text](https://github.com/mikeaadd/NOAA-Case-Study/img/hist_y.png "Histogram of Y")

![alt text](https://github.com/mikeaadd/NOAA-Case-Study/img/hist_logy.png "Histogram of log(Y)")

- point biserial of all predictors by target

<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th></th>\n      <th>Var</th>\n      <th>R</th>\n      <th>p</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>5</th>\n      <td>visibility_min</td>\n      <td>-0.631</td>\n      <td>0.0</td>\n    </tr>\n    <tr>\n      <th>15</th>\n      <td>humidity_mean</td>\n      <td>0.560</td>\n      <td>0.0</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>visibility_mean</td>\n      <td>-0.507</td>\n      <td>0.0</td>\n    </tr>\n    <tr>\n      <th>17</th>\n      <td>humidity_min</td>\n      <td>0.505</td>\n      <td>0.0</td>\n    </tr>\n    <tr>\n      <th>16</th>\n      <td>humidity_max</td>\n      <td>0.493</td>\n      <td>0.0</td>\n    </tr>\n    <tr>\n      <th>14</th>\n      <td>dew_point_tmpF_min</td>\n      <td>0.245</td>\n      <td>0.0</td>\n    </tr>\n    <tr>\n      <th>12</th>\n      <td>dew_point_tmpF_mean</td>\n      <td>0.243</td>\n      <td>0.0</td>\n    </tr>\n    <tr>\n      <th>27</th>\n      <td>pressure_change_mean</td>\n      <td>-0.234</td>\n      <td>0.0</td>\n    </tr>\n  </tbody>\n</table>

# Modeling

-  Ran three models to compare
    1. Logistic Regression
    2. Logistic Regression: Lasso (or l1)
    3. Logistic Regression: Ridge (or l2)

- Kfold Cross Validated (8 folds)

- mean results...

<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th></th>\n      <th>Log Regression</th>\n      <th>Log Regression(Lasso)</th>\n      <th>Log Regression(Ridge)</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>accuracy</th>\n      <td>0.885</td>\n      <td>0.884</td>\n      <td>0.882</td>\n    </tr>\n    <tr>\n      <th>auc</th>\n      <td>0.943</td>\n      <td>0.943</td>\n      <td>0.943</td>\n    </tr>\n    <tr>\n      <th>precision</th>\n      <td>0.776</td>\n      <td>0.776</td>\n      <td>0.774</td>\n    </tr>\n    <tr>\n      <th>recall</th>\n      <td>0.756</td>\n      <td>0.748</td>\n      <td>0.741</td>\n    </tr>\n  </tbody>\n</table>

- wet bulb and dew point were surprisingly predictive

<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th></th>\n      <th>Beta</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>wet_bulb_tmpF_max</th>\n      <td>-5.073</td>\n    </tr>\n    <tr>\n      <th>dew_point_tmpF_max</th>\n      <td>4.784</td>\n    </tr>\n    <tr>\n      <th>sea_lvl_pressure_mean</th>\n      <td>-1.410</td>\n    </tr>\n    <tr>\n      <th>dry_bulb_tmpF_max</th>\n      <td>1.285</td>\n    </tr>\n    <tr>\n      <th>dry_bulb_tmpF_mean</th>\n      <td>-1.252</td>\n    </tr>\n    <tr>\n      <th>pressure_min</th>\n      <td>1.167</td>\n    </tr>\n    <tr>\n      <th>visibility_min</th>\n      <td>-1.018</td>\n    </tr>\n    <tr>\n      <th>sea_lvl_pressure_max</th>\n      <td>0.890</td>\n    </tr>\n  </tbody>\n</table>

- logistic by month (controlling for seasonality)


<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th></th>\n      <th>Train</th>\n      <th>Test</th>\n      <th>Tampa</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>Apr</th>\n      <td>0.943</td>\n      <td>0.912</td>\n      <td>0.846</td>\n    </tr>\n    <tr>\n      <th>Aug</th>\n      <td>0.915</td>\n      <td>0.876</td>\n      <td>0.848</td>\n    </tr>\n    <tr>\n      <th>Dec</th>\n      <td>0.975</td>\n      <td>0.925</td>\n      <td>0.767</td>\n    </tr>\n    <tr>\n      <th>Feb</th>\n      <td>0.919</td>\n      <td>0.865</td>\n      <td>0.788</td>\n    </tr>\n    <tr>\n      <th>Jan</th>\n      <td>0.946</td>\n      <td>0.866</td>\n      <td>0.894</td>\n    </tr>\n    <tr>\n      <th>Jul</th>\n      <td>0.873</td>\n      <td>0.886</td>\n      <td>0.863</td>\n    </tr>\n    <tr>\n      <th>Jun</th>\n      <td>0.935</td>\n      <td>0.915</td>\n      <td>0.894</td>\n    </tr>\n    <tr>\n      <th>Mar</th>\n      <td>0.954</td>\n      <td>0.907</td>\n      <td>0.900</td>\n    </tr>\n    <tr>\n      <th>May</th>\n      <td>0.903</td>\n      <td>0.904</td>\n      <td>0.888</td>\n    </tr>\n    <tr>\n      <th>Nov</th>\n      <td>0.969</td>\n      <td>0.916</td>\n      <td>0.826</td>\n    </tr>\n    <tr>\n      <th>Oct</th>\n      <td>0.956</td>\n      <td>0.916</td>\n      <td>0.878</td>\n    </tr>\n    <tr>\n      <th>Sep</th>\n      <td>0.930</td>\n      <td>0.889</td>\n      <td>0.926</td>\n    </tr>\n    <tr>\n      <th>Overall</th>\n      <td>94.500</td>\n      <td>0.942</td>\n      <td>0.904</td>\n    </tr>\n  </tbody>\n</table>

# Final Thoughts

- use near by weather stations

- HLM
