# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model


start = '2010-01-01'
end = '2019-12-31'

st.title('Stock Trend prediction')
st.image('stock-trend.jpg')
user_input = st.text_input('Enter stock ticker', 'AAPL')
df = data.DataReader(user_input,'yahoo', start, end)


# Data Description
st.subheader('Data from 2010-2019')
st.write(df.describe())


# Data Visualization
st.subheader('Closing Price Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price Chart vs Time (with 100MA)')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
st.pyplot(fig)


st.subheader('Closing Price Chart vs Time (with 200MA)')
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
st.pyplot(fig)


data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
scaler = MinMaxScaler(feature_range=(0,1))
data_train_arr = scaler.fit_transform(data_train)

# Model Creation - Prebuilt optimized model
model = load_model('keras_model.h5')

# Testing Part
past_days = data_train.tail(100)
final_df = past_days.append(data_test, ignore_index=True)
input_data = scaler.fit_transform(final_df)


# Splitting Data into X_test & Y_test
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100: i])
  y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_pred = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor

# Final Visualization of our predicted model
st.subheader('Prediction VS Original')
final_fig = plt.figure(figsize=(12,6))
plt.plot(y_test, label='Original Price')
plt.plot(y_pred,'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(final_fig)