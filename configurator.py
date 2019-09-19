# -*- coding: utf-8 -*-

#
# Load the configuration data for machine learning
#
class ConfiguratorClass:
    def __init__(self, configPath):
        self._configPath = configPath
        self._separator = "="
        self._keys = {}
    
    def loadConfig(self):
        with open(self._configPath) as f:
            for line in f:
                if self._separator in line:
                    name, value = line.split(self._separator, 1)
                    self._keys[name.strip()] = value.strip()
        return 
    
    #
    # ratio is used to split data for training and testing
    # i.e. ratio=0.8, means 80% data for training, and 20%
    # for testing
    #
    def getSplitRate(self):
        return self._keys.get('split.rate')
    
    #
    # number of data used to generate time series array
    #    
    def getSeries(self):
        return self._keys.get('train.series')
    
    #
    # number of samples passed through to network
    #
    def getBatchSize(self):
        return self._keys.get('train.batch.size')
    
    #
    # epochs
    #
    def getEpochs(self):
        return self._keys.get('train.epochs')
    
    #
    # get loss func
    #
    def getLossFunction(self):
        return self._keys.get('train.loss.function')
    
    #
    # optimizer
    #
    def getOptimizer(self):
        return self._keys.get('train.optimizer')
    
