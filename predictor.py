# -*- coding: utf-8 -*-

#
# predictor price with trained model
#
class PredictorClass:
    def __init__(self, model):
        self._model = model
    
    #
    # predict based on input data
    #
    def predict(self, data):
        #print(data)
        return self._model.predict(data)
        
