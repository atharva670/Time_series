
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
df = pd.read_excel('AirPassengers.xlsx')
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
data = df['#Passengers']   
plt.plot(data)
plt.title("Time Series Data")
plt.show()
train = data[:int(len(data)*0.8)]
test = data[int(len(data)*0.8):]
model = ARIMA(train, order=(1,1,1))
model_fit = model.fit()
pred = model_fit.forecast(steps=len(test))
plt.plot(train, label='Train')
plt.plot(test, label='Actual')
plt.plot(pred, label='Predicted')
plt.legend()
plt.show() 