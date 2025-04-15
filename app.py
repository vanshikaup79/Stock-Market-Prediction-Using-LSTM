import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
model = load_model(r'D:\vs code programs\project_exhibition_1\Niftystockmodel.keras')
st.header('Nifty 50 Price Prediction Model')
data = yf.download('^NSEI', '2015-01-01', '2024-12-15')
data = pd.DataFrame(data)
data.reset_index(inplace=True)
if data.isnull().values.any():
    data.fillna(method='ffill', inplace=True)  
data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
data.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']


#Nifty50 data
st.subheader('Nifty 50 Price Data')
st.write(data)

#Nifty50 line chart
st.subheader('Nifty 50 Line Chart')
a=data.copy()
a = a[['Close']] 
st.line_chart(a)

#user inputtted year chart
selected_year = st.number_input('Select a year:', min_value=2015, max_value=2024, value=2023)
data['Year'] = pd.to_datetime(data['Date']).dt.year
filtered_data = data[data['Year'] == selected_year]
filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])
filtered_data['Date'] = filtered_data['Date'].dt.strftime('%Y-%m-%d') ####
st.subheader(f'Nifty 50 Price Data for {selected_year}')
st.write(filtered_data[['Date', 'Close']])
st.subheader(f'Nifty 50 Close Price Line Chart for {selected_year}')
st.line_chart(filtered_data.set_index('Date')['Close'])


#user inputted month chart
st.subheader('Select Month for Price Data')
selected_month = st.number_input('Select a month:', min_value=1, max_value=12, value=1)
filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])
monthly_filtered_data = filtered_data[filtered_data['Date'].dt.month == selected_month]
monthly_filtered_data['Date'] = monthly_filtered_data['Date'].dt.strftime('%Y-%m-%d')
st.subheader(f'Nifty 50 Price Data for {selected_year} - Month: {selected_month}')
st.write(monthly_filtered_data[['Date', 'Close']])
st.subheader(f'Nifty 50 Close Price Line Chart for {selected_year} - Month: {selected_month}')
st.line_chart(monthly_filtered_data.set_index('Date')['Close'])





data = data[['Close']] 

train_data = data[:-50]
test_data = data[-50:]

scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = scaler.fit_transform(train_data)

test_data_scaled = scaler.transform(test_data)


base_days = 10
future_days = st.number_input('Select number of days for prediction:', min_value=1, max_value=30, value=5)
x_train, y_train = [], []
for i in range(base_days, train_data_scaled.shape[0]):
    x_train.append(train_data_scaled[i-base_days:i])
    y_train.append(train_data_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


pred = model.predict(x_train)
pred = pred.reshape(-1, 1)

pred = scaler.inverse_transform(pred)

preds = pred.reshape(-1, 1)
ys = scaler.inverse_transform(y_train.reshape(-1, 1))
preds = pd.DataFrame(preds, columns=['Predicted Price'])
ys = pd.DataFrame(ys, columns=['Original Price'])
chart_data = pd.concat((preds, ys), axis=1)


last_sequence = x_train[-1:]  
future_predictions = []

#pr4diction for future days
for i in range(future_days):
    pred = model.predict(last_sequence, verbose=0)
    future_predictions.append(pred[0][0])
    new_sequence = last_sequence[0]
    new_sequence = np.roll(new_sequence, -1)
    new_sequence[-1] = pred[0][0]
    last_sequence = new_sequence.reshape(1, base_days, 1)

future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions = scaler.inverse_transform(future_predictions)

st.subheader(f'Future Nifty 50 Prices for Next {future_days} Days')
future_predictions_df = pd.DataFrame(future_predictions, columns=['Future Price'])
st.write(future_predictions_df)
