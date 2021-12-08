#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import pickle


data = pd.read_csv("C:\\Users\\Mateen\\Expenses_prediction.csv")

data.shape

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
data_scaled = scaler.fit_transform(data)
df = pd.DataFrame.from_records(data_scaled)


X = df.iloc[:,:2]


Y= df.iloc[:,-1]


from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X, Y)


from sklearn.metrics import r2_score
Y_predict = lr_model.predict(X)
r2 = r2_score(Y, Y_predict)
print('R2 score is {}'.format(r2))

# Saving model to disk
pickle.dump(lr_model, open('C:\\Users\\Mateen\\lr_model.pkl','wb'))
