import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')

class BackTest:

	def __init__(self,sarima=True):
		"""
		Initialize the class
		sarima: use sarima for backtesting model
		"""
		self.mse = []

	def __repr__(self):
		return ('backtesting tool')

	def backtest(self,df):
		"""
		backtest fuction
		"""
		length = len(df)
		# mse = []
		x = 1
		while x < len(df):
			train = df.iloc[:length - x]
			test = df.iloc[length - x:length]

			model = auto_arima(df,seasonal=True,m=12,error_action='ignore',suppress_warnings=True)
			model.fit(train)
			future_forecast = model.predict(n_periods = len(test))

			#Plot the prediction against the test
			plt.plot(np.array(test),color='blue',label='Test')
			plt.plot(future_forecast,color='red',label='Prediction')
			plt.legend()
			plt.show()

			#Evaluation Metric
			error = mean_squared_error(test,future_forecast)
			self.mse.append(round(error,4))
			x+=5
	def MeanSqError(self):
		"""
		Returns list of MSE
		"""
		return self.mse