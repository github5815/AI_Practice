# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#
# Prepare the dataset for training data model
#
class DataProcClass:
    
    def __init__(self, csvFullPath, configurator):
        self._configurator = configurator
        self._csvPath = csvFullPath
        self._scaler = MinMaxScaler(feature_range=(0,1))
        self._isReshaped = False
        self._data = None
        self._normData = None
        self._trainData = None
        self._testData = None
    
    #dataframe types
    #date       object
    #close     float64
    #volume     object
    #open      float64
    #high      float64
    #low       float64
    #dtype: object
    def loadData(self):
        df = pd.read_csv(self._csvPath)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        return df
    
    #
    # input as series, output is numpy.narray
    #
    def reshape(self, dataFrame):
        close = dataFrame['close']  # get close as Series data
        close = close.values.reshape(len(close), 1)
        self.isReshaped = True
        return close
    
    #
    # get data as numpy array
    #
    def getData(self):
        if not isinstance(self._data, np.ndarray):
            df = self.loadData()
            self._data = self.reshape(df)
        return self._data
    
    #
    # normalize dataset
    #
    def getNormalizedDs(self):
        if not isinstance(self._normData, np.ndarray):
            self._normData = self._scaler.fit_transform(self.getData())
        return self._normData

    #
    # get trainin data
    #
    def getTrainData(self):
        if not isinstance(self._trainData, np.ndarray):
           self.sliceTrainingData()
        return self._trainData

    #
    # get test data
    #
    def getTestData(self):
        if not isinstance(self._testData, np.ndarray):
            self.sliceTrainingData()
        return self._testData

    #
    #  slice dataset into training data and test data
    #        
    def sliceTrainingData(self):
        self.getNormalizedDs()
        splitRate = float(self._configurator.getSplitRate())
        train_size = int(len(self._normData) * splitRate)
        
        self._trainData, self._testData = self._normData[0:train_size, :], self._normData[train_size:len(self._normData), :]
        print('split data into train and test:', len(self._trainData), len(self._testData))
    
    #
    #  get scaler for normalize and denormalize
    #    
    def getScaler(self):
        return self._scaler
    
    #
    # denormalize the data
    #
    def denormalize(self, data):
        return self._scaler.inverse_transform(data)
    
    