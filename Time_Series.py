
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import math
df = pd.read_excel('AirPassengers.xlsx')
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
df=df.asfreq('MS')
print('Current Prediction')
print(df)
data = df['#Passengers']   
model = ARIMA(df, order=(1,1,1))
model_fit = model.fit()
pred = model_fit.forecast(steps=12)
pred=pred.astype('int64')
print('Future Prediction\n')
print(pred)
print(type(pred))
plt.plot(pred,color='red',label='Date vs Values',marker='o')
plt.legend()
plt.xlabel('Current')
plt.ylabel('Forecasted')
plt.show()
train=data[:int(len(data)*0.8)]
test=data[int(len(data)*0.8):]
model1 = ARIMA(train, order=(1,1,1))
model_fit1 = model1.fit()
pred1 = model_fit1.forecast(steps=len(test))
pred1=pred1.astype('int64')
df4=pd.DataFrame({'Actual':test,'Predicted':pred1})
print(df4)
print('MAE',mean_absolute_error(pred1,test))
print('MSE',mean_squared_error(pred1,test))
print('RMSE',math.sqrt(mean_absolute_error(pred1,test)))
print('R2 Score',r2_score(test,pred1))




