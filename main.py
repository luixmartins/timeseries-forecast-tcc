import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import warnings 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.svm import SVR 

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

        return self.df 
    
    def modeling_for_forecast(self): 
        self.df['target'] = self.df['close'].shift(periods=self.timestep)
        self.df.drop(['close'], axis=1, inplace=True)
        self.df.dropna(inplace=True)

        return self.df 

    def make_forecast(self):
        self.preprocessing_data()
        self.modeling_for_forecast()

pipeline = Pipeline(path='dataset/soybean_bovespa.csv')

df = pipeline.preprocessing_data()

print(df.tail())