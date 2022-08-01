import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import warnings 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.svm import SVR 
from sklearn.metrics import mean_squared_error, mean_absolute_error

class Pipeline:
    def __init__(self, path:str, kernel:str='poly', timesteps:int=5) -> None:
        self.df = pd.read_csv(path)
        self.model = SVR(kernel=kernel)
        self.timestep = timesteps 

    def replace_values(self, value):
        if value == '-':
            return np.nan 
        
        characters = {
            ',': '.',
            'K': '',
            '%': '',
        }
        return value.translate(str.maketrans(characters))

    def preprocessing_data(self):
        self.df.columns = ['date', 'close', 'open', 'high', 'low', 'vol', 'change']
        self.df['date'] = pd.to_datetime(self.df['date'], format="%d.%m.%Y")

        #Apply replace values function 
        self.df['close'] = self.df['close'].apply(lambda value: self.replace_values(value))
        self.df['open'] = self.df['open'].apply(lambda value: self.replace_values(value))
        self.df['high'] = self.df['high'].apply(lambda value: self.replace_values(value))
        self.df['low'] = self.df['low'].apply(lambda value: self.replace_values(value))
        self.df['vol'] = self.df['vol'].apply(lambda value: self.replace_values(value))
        self.df['change'] = self.df['change'].apply(lambda value: self.replace_values(value))

        #Remove NaN values 
        self.df.fillna(method='ffill', inplace=True)

        #Create index with datetime 
        self.df.index = self.df.pop('date')

        #Reverse dataset 
        self.df = self.df.iloc[::-1]
        
        #Convert values to float 
        self.df[self.df.columns] = self.df[self.df.columns].astype('float64')

        self.df = self.modeling_for_forecast()

        return self.df 
    
    def modeling_for_forecast(self): 
        self.df['target'] = self.df['close'].shift(periods=self.timestep)
        self.df.drop(['close'], axis=1, inplace=True)
        self.df.dropna(inplace=True)

        return self.df 

    def make_forecast(self):
        self.preprocessing_data()

        df_train = self.df.iloc[:-5]
        df_test = self.df.iloc[-5:]

        X_train, y_train = df_train[['open', 'high', 'low', 'vol', 'change']], df_train['target']
        X_test, y_test = df_test[['open', 'high', 'low', 'vol', 'change']], df_test['target']
        
        #X_train_scaled, y_train_scaled = MinMaxScaler((-1, 1)).fit_transform(X_train), MinMaxScaler((-1, 1)).fit_transform(y_train.values.reshape(1, -1))
        #X_test_scaled = MinMaxScaler((-1, 1)).fit_transform(X_test)

        #y_scaler = MinMaxScaler((-1, 1)).fit(y_test.values.reshape(1, -1))

        model = SVR(kernel="linear", C=1.5, epsilon=0.15)
        model.fit(X_train.values, y_train.values)

        y_pred = model.predict(X_test.values)

        #y_pred_inverse = y_scaler.inverse_transform(y_pred.reshape(1, -1))

        print("\nMSE: ", mean_squared_error(y_test.values, y_pred))
        print("MAE: ", mean_absolute_error(y_test.values, y_pred))
        print("\nValores previstos: ", y_pred.reshape(-1), "\n")

pipeline = Pipeline(path='dataset/soybean_bovespa.csv')

df = pipeline.make_forecast()