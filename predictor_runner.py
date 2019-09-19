# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error

import configurator
import dataset_preparator
import trainer
import predictor

#
#   Initialize the dataset properator
#
def get_ds_preparator(cvs_path, config):
    ds_preparator = dataset_preparator.DataProcClass(cvs_path, config)
    return ds_preparator

#
#   start the training process
#
def start_trainer(ds_preparator, config):
    ml = trainer.TrainerClass(config)
    ml.run(ds_preparator.getTrainData(), ds_preparator.getTestData())
    return ml
    
#
#   validatate the model performance
#
def validate(ds_preparator, ml):
    #validate predictor performance
    #validator = predictor.PredictorClass(ml.getModel(), ml.getSeries())
    validator = predictor.PredictorClass(ml.getModel())
    
    train_x, train_y = ml.createTS(ds_preparator.getTrainData())
    predict_train_data = validator.predict(train_x)
    
    test_x, test_y = ml.createTS(ds_preparator.getTestData())
    predict_test_data = validator.predict(test_x)
    
    # denormalize predict data
    predict_train_data = ds_preparator.denormalize(predict_train_data)
    predict_test_data = ds_preparator.denormalize(predict_test_data)
    
    predict_display(ml.getSeries(), ds_preparator.getData(), predict_train_data, predict_test_data)
    
    # calculate trainning score
    train_y = ds_preparator.denormalize([train_y])
    test_y = ds_preparator.denormalize([test_y])
    
    train_score = math.sqrt(mean_squared_error(train_y[0], predict_train_data[:, 0]))
    test_score = math.sqrt(mean_squared_error(test_y[0], predict_test_data[:, 0]))
    
    return train_score, test_score

#
#   utility for display one set data
#
def display(data):
    plt.plot(data)
    plt.show()

#
#  utility for display orignal data, predict data
#
def predict_display(series, orig_data, train_predict_data, test_predict_data):
    train_predict_plot = np.empty_like(orig_data)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[series:len(train_predict_data) + series, :] = train_predict_data
    
    test_predict_plot = np.empty_like(orig_data)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict_data) + (series * 2) + 1:len(orig_data)-1, :] = test_predict_data
    
    #plot on graph
    plt.plot(orig_data)
    plt.plot(train_predict_plot)
    plt.plot(test_predict_plot)
    plt.show()

#
#  main entry for starting training network and predicting process
#    
def main(cvs_path, config_path):
    # get configurator
    config = configurator.ConfiguratorClass(config_path)
    config.loadConfig()
    
    #prepare dataset for training
    ds_preparator = get_ds_preparator(cvs_path, config)
    display(ds_preparator.getNormalizedDs())
    
    #prepare training model
    ml = start_trainer(ds_preparator, config)
    
    #validate the result
    train_score, test_score = validate(ds_preparator, ml)
    print('train rmse: %.2f' % train_score)
    print('test rmse: %0.2f' % test_score)
    
    
if __name__ == "__main__":
    cvs_path = "c:\\work\\ai\\stock\\data\\MSFT.csv"
    config_path = "c:\\work\\ai\\stock\\stock\\config.properties"
    main(cvs_path, config_path)