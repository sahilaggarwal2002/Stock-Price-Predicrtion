import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dropout,Dense
import tensorflow as tf
import matplotlib.pyplot as plt

import yfinance as yf
from yahoofinancials import YahooFinancials



stock=input("Enter the name of stock ")
df = yf.download(stock, start="2020-01-19", end="2022-01-17")
df.to_csv('main.csv')
df = pd.read_csv('main.csv')
print(df)
dependent_var=df["Close"]
plt.plot(dependent_var)


#normalize
scaler=MinMaxScaler(feature_range=(0,1))
dependent_var=scaler.fit_transform(np.array(dependent_var).reshape(-1,1))


#TRAIN AND TEST

testlength=int(0.7*(len(dependent_var)))

train=dependent_var[:testlength]
test=dependent_var[testlength:]
test2=scaler.inverse_transform(test)



#preprocessing

def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)
timestep=100
Xtrain, Ytrain= create_dataset(train,timestep)
Xtest, Ytest= create_dataset(test,timestep)

#converting into 3dimension
Xtrain=Xtrain.reshape(Xtrain.shape[0],Xtrain.shape[1],1)
Xtest=Xtest.reshape(Xtest.shape[0],Xtest.shape[1],1)

#Stacked LSTM
lstm_model=Sequential()
lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(Xtrain.shape[1],1)))
lstm_model.add(LSTM(units=50,return_sequences=True))
lstm_model.add(LSTM(50))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error',optimizer='adam')
lstm_model.fit(Xtrain,Ytrain,epochs=100,batch_size=64,verbose=1)

#prediction
train_predict=lstm_model.predict(Xtrain)
test_predict=lstm_model.predict(Xtest)

#Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)
plt.plot(train_predict)
plt.plot(test_predict)
plt.show()



look_back=100
trainPredictPlot = np.empty_like(dependent_var)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

testPredictPlot = np.empty_like(dependent_var)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(dependent_var)-1, :] = test_predict

plt.plot(scaler.inverse_transform(dependent_var))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

import os
os.remove("main.csv")
