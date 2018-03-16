# NOAA-Case-Study: Will It Rain?

1. Does Temperature, sunshine, humidity, wind direction and air pressure predict precipitation in the Denver Airport Weather Station

2. If Temperature, sunshine, humidity, wind direction and air pressure does predict precipitation which predictor is the most significant

# The Data

- hourly weather data from the Denver Airport Weather Station from 1989 to 2018

- 358000 rows and 19 columns

- Variable Info!

|    | Variable         | dtype          | Description                                                                                                           |
|---:|:-----------------|:---------------|:----------------------------------------------------------------------------------------------------------------------|
|  0 | station_name     | object         | Date of Observation to the minute                                                                                     |
|  1 | datetime         | datetime64[ns] | Code for type of station making observation                                                                           |
|  2 | report_type      | object         | Cloud Type indicating percent coverage                                                                                |
|  3 | sky_conditions   | int64          | Horizontal Distance an object can be seen                                                                             |
|  4 | visibility       | float64        | Code for type of station making observation                                                                           |
|  5 | weather_type     | object         | A series of codes summarizing the rest of the data                                                                    |
|  6 | dry_bulb_tmpF    | float64        | Standard Air Temperature Reported                                                                                     |
|  7 | wet_bulb_tmpF    | float64        | Temperature subtracted by dewpoint depression                                                                         |
|  8 | dew_point_tmpF   | float64        | Temperature at which air can no longer hold all of its water vapor                                                    |
|  9 | humidity         | float64        | This is the relative humidity given to the nearest whole percentageSpeed of the wind at the time of observation (mph) |
| 10 | wind_speed       | float64        | Wind direction from true north using compass directions (degrees)                                                     |
| 11 | gust_speed       | float64        | Wind gusts occurring during time of observation                                                                       |
| 12 | pressure         | float64        | Atmospheric pressure observed at the station during the time of observation. (in Hg)                                  |
| 13 | pressure_tedency | float64        | Pressure tendency of the last few hours                                                                               |
| 14 | pressure_change  | float64        | Difference in pressure over the past 3 hours.                                                                         |
| 15 | sea_lvl_pressure | float64        | Sea level pressure in Hg                                                                                              |
| 16 | precipitation    | float64        | Amount of precipitation in inches to hundredths over the past hour.                                                   |

- missing data in primarily gust_speed and weather type

|                  |   Total |   Percent |
|:-----------------|--------:|----------:|
| gust_speed       |  310749 |     0.868 |
| weather_type     |  301449 |     0.842 |
| pressure_change  |  238747 |     0.667 |
| pressure_tedency |  204502 |     0.571 |
| precipitation    |  126642 |     0.354 |
| wet_bulb_tmpF    |   79204 |     0.221 |
| pressure         |   79111 |     0.221 |
| sea_lvl_pressure |   43019 |     0.12  |

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

![heat map](https://github.com/mikeaadd/NOAA-Case-Study/raw/master/img/corr_heat.png "Correlation Heat Map")

- Histogram of target Variable, precipitation

![Hist Y](https://github.com/mikeaadd/NOAA-Case-Study/raw/master/img/hist_y.png "Histogram of Y")

![Hist LogY](https://github.com/mikeaadd/NOAA-Case-Study/raw/master/img/hist_logy.png "Histogram of log(Y)")

- point biserial of all predictors by target

|    | Var                  |      R |   p |
|---:|:---------------------|-------:|----:|
|  0 | visibility_min       | -0.631 |   0 |
|  1 | humidity_mean        |  0.56  |   0 |
|  2 | visibility_mean      | -0.507 |   0 |
|  3 | humidity_min         |  0.505 |   0 |
|  4 | humidity_max         |  0.493 |   0 |
|  5 | dew_point_tmpF_min   |  0.245 |   0 |
|  6 | dew_point_tmpF_mean  |  0.243 |   0 |
|  7 | pressure_change_mean | -0.234 |   0 |

# Modeling

-  Ran three models to compare
    1. Logistic Regression
    2. Logistic Regression: Lasso (or l1)
    3. Logistic Regression: Ridge (or l2)

- Kfold Cross Validated (8 folds)

- mean results...

|                       |   accuracy |   auc |   precision |   recall |
|:----------------------|-----------:|------:|------------:|---------:|
| Log Regression        |      0.888 | 0.945 |       0.787 |    0.757 |
| Log Regression(Lasso) |      0.886 | 0.945 |       0.786 |    0.745 |
| Log Regression(Ridge) |      0.885 | 0.945 |       0.783 |    0.745 |

- wet bulb and dew point were surprisingly predictive

|                       |   Beta |
|:----------------------|-------:|
| wet_bulb_tmpF_max     | -5.073 |
| dew_point_tmpF_max    |  4.784 |
| sea_lvl_pressure_mean | -1.41  |
| dry_bulb_tmpF_max     |  1.285 |
| dry_bulb_tmpF_mean    | -1.252 |
| pressure_min          |  1.167 |
| visibility_min        | -1.018 |
| sea_lvl_pressure_max  |  0.89  |


# Final Results!

- logistic by month (controlling for seasonality

|         |   Train |   Test |   Tampa |
|:--------|--------:|-------:|--------:|
| Jan     |   0.946 |  0.866 |   0.894 |
| Feb     |   0.919 |  0.865 |   0.788 |
| Mar     |   0.954 |  0.907 |   0.9   |
| Apr     |   0.943 |  0.912 |   0.846 |
| May     |   0.903 |  0.904 |   0.888 |
| Jun     |   0.935 |  0.915 |   0.894 |
| Jul     |   0.873 |  0.886 |   0.863 |
| Aug     |   0.915 |  0.876 |   0.848 |
| Sep     |   0.93  |  0.889 |   0.926 |
| Oct     |   0.956 |  0.916 |   0.878 |
| Nov     |   0.969 |  0.916 |   0.826 |
| Dec     |   0.975 |  0.925 |   0.767 |
| Overall |  94.5   |  0.942 |   0.904 |

# Final Thoughts

- Take into account near by weather stations

- Try to further control for seasonality
