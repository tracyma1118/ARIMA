

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib as m
import statsmodels
import statsmodels.api as sm
import matplotlib.pyplot as plt

df=pd.read_csv('.\\raw-all-noise-complaints.csv')
df.head()

df.rename(columns={'Received Date': 'Date', 
                   'Complaint Sub Type': 'Subtype',
                   'Received Time':'Time'    }, inplace=True)

df['Date']= pd.to_datetime(df['Date'])

df = df[(df['Date'] >= '2022-01-01') & (df['Date'] < '2023-04-30')]


df.to_csv("2022-202307data.csv")

df = df.drop(df.columns[[0, 2, 3,6,7,8,9,10]], axis=1) 
df = df.drop(df.columns[[1,2]], axis=1)  



#------------------------------------------------------------------
df1 = pd.read_csv(".//2020-2022data.csv")
df1.set_index('Unnamed: 0',inplace=True)
df1 = df1.drop(df1.columns[[0, 2, 3,6,7,8,9,10,11,12,13]], axis=1) 
df1 = df1.drop(df1.columns[1], axis=1)  
df1 = df1.reset_index(drop=True) 
df1['Volume']='1'
df1['Volume'] = df1.groupby(['Date'])['Volume'].transform('count')
df1 = df1.sort_values(by=['Date'])
#df1['Date']= pd.to_datetime(df1['Date'])
#df1 = df1.set_index('Date')

df1 = df1[(df1['Date'] >= '2022-01-01') & (df1['Date'] < '2023-07-01')]
df1 = df1.drop_duplicates(subset = ['Date'], keep ='last')
##-------------------------------------------------------------------


from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#def read_data():
    #df1 = pd.read_csv('.//2020-2022data.csv')
    #return df1

def convert_df_to_timeseries(df1):
    df1 = df1.groupby('Date').size().reset_index(name='Volume')
    return df1


df = convert_df_to_timeseries(df)

def preprocess_df(df1):

    print(f'WARNING: Converting date column to datetime')
    df1['Date'] = pd.to_datetime(df1['Date'])
    df1['Date'] = df1['Date'].dt.date

    print(f'WARNING: Dropping hardcoded date:')
    date_to_drop = pd.to_datetime('2023-07-01')
    print(date_to_drop)
    df1.drop(df1[df1['Date'] == date_to_drop].index, inplace=True)

    return df1


df = preprocess_df(df)
df1.dtypes
df1 = df1.sort_values(by=['Date'])

def apply_MAD_filter(df1):

    print(f'WARNING: Applying MAD Filter')

    num_rows_before = len(df1)

# Aggressive to get rid of the 1 case with huge >100 spike
    mad_threshold_percentile = 0.5  
    median = df1['Volume'].median()
    mad = (df1['Volume'] - df1['Volume'].mean()).abs().mean()
    mad_threshold = mad * pd.Series.quantile(pd.Series.abs(df1['Volume'] - median), mad_threshold_percentile)
    df_filtered = df1[
    (df1['Volume'] >= median - mad_threshold) & (df1['Volume'] <= median + mad_threshold)]


# Log the number of rows after filtering
    num_rows_after = len(df_filtered)

# Log the number of rows affected
    num_rows_affected = num_rows_before - num_rows_after
    print(f'Number of rows affected by MAD filter: {num_rows_affected}')

    return df_filtered




def do_lineplot(df1):
    plt.figure(figsize=(12, 6))
    plt.plot(df1['Date'], df1['Volume'], marker='.')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.title('Complaints Per Day')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return


do_lineplot(df)

df = apply_MAD_filter(df)
do_lineplot(df)              #observe the difference after medain filter

def perform_adf_test(timeseries):
    result = adfuller(timeseries)
    print("Stationarity Test for Original Data:")
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')
    return result

# Test if differencing is required 
adf_result = perform_adf_test(df['Volume'])



def do_ACF_PACF_plots(timeseries):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plot_acf(df['Volume'], lags=30, ax=plt.gca())
    plt.subplot(2, 1, 2)
    plot_pacf(df['Volume'], lags=30, ax=plt.gca())
    plt.tight_layout()
    plt.show()
    return

do_ACF_PACF_plots(df)


# required datetime index
df.set_index('Date', inplace=True) 
df.index.freq = 'D'  

# autoregressive lags, differencing, and moving average lags
p = 1
d = 0
q = 1


# seasonal autoregressive lags, seasonal differencing, and seasonal moving average lags
seasonal_p = 1
seasonal_d = 0
seasonal_q = 0
seasonal_period = 6

# Create the ARIMA model with seasonal components
model = ARIMA(df['Volume'], order=(p, d, q), seasonal_order=(seasonal_p, seasonal_d, seasonal_q, seasonal_period))
result = model.fit()

# Forecasting
forecast_horizon = 30  # Replace this with the number of periods you want to forecast
forecast = result.forecast(steps=forecast_horizon)


plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Volume'], marker='o', label='Actual')
plt.plot(pd.date_range(start=df.index[-1], periods=forecast_horizon+1, closed='right'), forecast, marker='o', linestyle='dashed', label='Forecast')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title('ARIMA Forecast')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



## HERE: take a subset y_true vector to measure performance (i.e. use historical data we have as a test set)
# Actual values for comparison (replace y_true with the actual values for the forecast horizon)


from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_error 


y_true = df['Volume'][-30:]


mae = mean_absolute_error(y_true, forecast)
mse = mean_squared_error(y_true, forecast)
rmse = mean_squared_error(y_true, forecast, squared=False)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
