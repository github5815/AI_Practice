# -*- coding: utf-8 -*-

import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

#
# Setup the trainning model, and train the model
#
class TrainerClass:
    def __init__(self, config):
        print('Trainer Class')
        self._config = config
        self._series = int(config.getSeries())
    
    #create time series for training assuming series = 7
    # assuming arrData = [a0,a1,a2,a3,a4,...a6,a7, a8, a9... ]
    # X would be input
    # [[a0, a1, .., a6, a7],
    #  [a1, a2, .., a7, a8],
    #  ....
    # Y would be the output of X used for training
    # [a7, a8, a9, ..]                
    def createTS(self, arrData):
        X, Y = [], []
        for i in range(len(arrData) - self._series -1):
            item = arrData[i:(i + self._series)]
            X.append(item)
            Y.append(arrData[i + self._series, 0])
        return np.array(X), np.array(Y)
    
    #
    # start trainer
    #
    def run(self, trainData, testData):
        #create time series data for training
        trainX, trainY = self.createTS(trainData)
        testX, testY = self.createTS(testData)
        
        #reshape into LSTM format - samples, steps, features
        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
        testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
        
        self._trainX = trainX
        self._trainY = trainY
        
        self._testX = testX
        self._testY = testY
        
        #build the model
        self._model = Sequential()
        self._model.add(LSTM(4, input_shape=(self._series,1)))
        self._model.add(Dense(1))
        self._model.compile(loss=self._config.getLossFunction(), optimizer=self._config.getOptimizer())
        #fit the model
        self._model.fit(trainX, trainY, epochs=int(self._config.getEpochs()), batch_size=int(self._config.getBatchSize()))
        
    def getSeries(self):
        return self._series
    
    def getModel(self):
        return self._model
    
