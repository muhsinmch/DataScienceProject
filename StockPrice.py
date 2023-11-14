import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Fetch historical stock data from Yahoo Finance
symbol = 'ADANIGREEN.BO'  # Adani Green Energy Limited (you can change this to any stock symbol)
start_date = '2020-01-01'
end_date = '2022-01-01'

df = yf.download(symbol, start=start_date, end=end_date)

# Use 'Close' prices for prediction
data = df['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences and labels for training
def create_sequences(data, seq_length):
    sequences = []
    labels = []

    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = data[i+seq_length:i+seq_length+1]
        sequences.append(seq)
        labels.append(label)

    return np.array(sequences), np.array(labels)

# Set sequence length (number of days to look back)
seq_length = 10

sequences, labels = create_sequences(scaled_data, seq_length)

# Split data into training and testing sets
split_ratio = 0.8
split_index = int(split_ratio * len(sequences))

train_sequences = sequences[:split_index]
train_labels = labels[:split_index]
test_sequences = sequences[split_index:]
test_labels = labels[split_index:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(train_sequences, train_labels, epochs=50, batch_size=32)

# Evaluate the model on the test set
loss = model.evaluate(test_sequences, test_labels)
print(f'Test Loss: {loss}')

# Make predictions on the test set
predictions = model.predict(test_sequences)

# Inverse transform the predictions and test labels to the original scale
predictions = scaler.inverse_transform(predictions)
actual_prices = scaler.inverse_transform(test_labels)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(predictions, label='Predicted Prices')
plt.plot(actual_prices, label='Actual Prices')
plt.title('Stock Price Prediction using LSTM')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
