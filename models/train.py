import pandas as pandas
import numpy as np
From sklearn.model_selection import train_test_split
From sklearn.processing import StandardScaler
From tensorflow.keras.models import Sequential
From tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

df = pd.read_csv('data/customer_churn.csv')
x = df.drop(columns=['customerID', 'Surname','Exited'], axis=1)
y = df['Exited']
x['gender'] = label_encoder.fit_transform(x['gender'])
x['Geography'] = label_encoder.fit_transform(x['Geography'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)