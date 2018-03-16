# NOAA-Case-Study: Will It Rain?

1. Does Temperature, sunshine, humidity, wind direction and air pressure predict precipitation in the Denver Airport Weather Station

2. If Temperature, sunshine, humidity, wind direction and air pressure does predict precipitation which predictor is the most significant

# The Data

- hourly weather data from the Denver Airport Weather Station from 1989 to 2018

- 35800 rows and 19 columns

----------------  ------  --------
gust_speed        310749  0.867784
weather_type      301449  0.841813
pressure_change   238747  0.666714
pressure_tedency  204502  0.571083
precipitation     126642  0.353655
----------------  ------  --------



2. cleaned up data (mostly recoding object vars that were actually numeric varcs)

3. performed some eda: cor matrix, point-biserial for each continuous var

4. ran logistic regression

# Roadmap

1. compare to baseline model and model with extra lag variables so model can see further in past

...

...

2. use final model to get metrics on testing data

3. compare final model to data from a completely different station
    - does model fit data to a specific station or is it more generalizable?
