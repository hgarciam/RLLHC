import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import sys
import statsmodels.api as sm
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from multiprocessing import Pool
from collections import OrderedDict
from sklearn import linear_model
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
#import xgboost as xgb
#from sklearn.externals import joblib

class RLLHC(object):

    #def __init__(self):

    def RFmodel(self,n_estimators, max_depth, min_samples_leaf, min_samples_split, train_inputs,train_outputs):
        self.estimator = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth,min_samples_split=min_samples_split,random_state=0)
        self.estimator.fit(train_inputs, train_outputs)
        return self.estimator
    
    def mymodel(self,train_inputs,train_outputs,bagging):
        if bagging:
            ridge = linear_model.Ridge(normalize=False, tol=1e-50, alpha=1e-03)
            self.estimator = BaggingRegressor(base_estimator=ridge, n_estimators=10, max_samples=0.9, max_features=1.0, n_jobs=16, verbose=3)
        else:
            self.estimator = linear_model.Ridge(normalize=True, tol=1e-50, alpha=1e-03, random_state=0, solver='auto')

        self.estimator.fit(train_inputs, train_outputs)

        # joblib.dump(estimator, 'estimator.pkl')
        #self.estimator = joblib.load('estimator.pkl')
        return self.estimator
    
    def myXGBoost(self, train_inputs, train_outputs):
        self.estimator = xgb.XGBRegressor(verbosity=0)
        self.estimator.fit(train_inputs,train_outputs)
        return self.estimator

    def get_cross_validation(self, train_inputs, train_outputs):
        return cross_val_score(self.estimator, train_inputs, train_outputs, cv=5)

    def get_model_score(self,estimator,train_inputs,train_outputs,test_inputs,test_outputs):
        training_score = self.estimator.score(train_inputs, train_outputs)
        test_score = self.estimator.score(test_inputs, test_outputs)
        prediction_train = self.estimator.predict(train_inputs)
        mae_train = mean_absolute_error(train_outputs, prediction_train)
        prediction_test = self.estimator.predict(test_inputs)
        mae_test = mean_absolute_error(test_outputs, prediction_test)

        print("Training: R2 = {0}, MAE = {1}".format(training_score, mae_train))
        print("Test: R2 = {0}, MAE = {1}".format(test_score, mae_test))

    def clean_data(self,mqt_errors_b1, mqt_errors_b2):
        # Solving missing data
        # TODO check why some data in mqt_b1 is missing
        missing = 0
        for i in range(len(mqt_errors_b1)):
            if len(mqt_errors_b1[i])!=2:
                mqt_errors_b1[i] = mqt_errors_b2[i]
                missing += 1
            if len(mqt_errors_b2[i])!=2:
                mqt_errors_b2[i] = mqt_errors_b1[i]
                missing += 1
        #print("Missing MQT samples {}".format(missing))
        return mqt_errors_b1, mqt_errors_b2
    
    def get_feature_importance(self,estimator,mode):
                
                importance = estimator.coef_

                error_label = [
                    'Q3BL', 'Q3AL', 'Q2BL', 'Q2AL', 'Q1BL', 'Q1AL',
                    'Q1AR', 'Q1BR', 'Q2AR', 'Q2BR', 'Q3AR', 'Q3BR',
                    'MQT_1', 'MQT_2', 'MQT_3', 'MQT_4'
                    ]
                
                betas_label = [
                    r'$\beta_x$ IP1L B1',r'$\beta_x$ IP1R B1',r'$\beta_x$ IP2L B1',r'$\beta_x$ IP2R B1',\
                    r'$\beta_x$ IP5L B1',r'$\beta_x$ IP5R B1',r'$\beta_x$ IP8L B1',r'$\beta_x$ IP8R B1',\
                    r'$\beta_y$ IP1L B1',r'$\beta_y$ IP1R B1',r'$\beta_y$ IP2L B1',r'$\beta_y$ IP2R B1',\
                    r'$\beta_y$ IP5L B1',r'$\beta_y$ IP5R B1',r'$\beta_y$ IP8L B1',r'$\beta_y$ IP8R B1',\
                    r'$\beta_x$ IP1L B2',r'$\beta_x$ IP1R B2',r'$\beta_x$ IP2L B2',r'$\beta_x$ IP2R B2',\
                    r'$\beta_x$ IP5L B2',r'$\beta_x$ IP5R B2',r'$\beta_x$ IP8L B2',r'$\beta_x$ IP8R B2',\
                    r'$\beta_y$ IP1L B2',r'$\beta_y$ IP1R B2',r'$\beta_y$ IP2L B2',r'$\beta_y$ IP2R B2',\
                    r'$\beta_y$ IP5L B2',r'$\beta_y$ IP5R B2',r'$\beta_y$ IP8L B2',r'$\beta_y$ IP8R B2',\
                ]

                sns.heatmap(abs(importance), cmap="viridis", cbar=True)

                if mode == 'predictor':
                    plt.xticks(np.arange(0.5,32.5,1), betas_label, rotation=0)
                    plt.yticks(np.arange(0.5,16.5,1), error_label, rotation=0)
                else:
                    plt.yticks(np.arange(0.5,32.5,1), betas_label, rotation=0)
                    plt.xticks(np.arange(0.5,16.5,1), error_label, rotation=0)                    

                plt.show()


    def norm_beta(self,beta_vector_x1,beta_vector_y1,beta_vector_x2,beta_vector_y2):
        beta_norm_x1 = []
        beta_norm_y1 = []
        beta_norm_x2 = []
        beta_norm_y2 = []
        beta_x_b1_nominal = [1592.13, 1592.15, 1, 1, 1, 1, 1, 1]
        beta_y_b1_nominal = [1592.13, 1592.15, 1, 1, 1, 1, 1, 1]
        beta_x_b2_nominal = [1592.14, 1592.13, 1, 1, 1, 1, 1, 1]
        beta_y_b2_nominal = [1592.14, 1592.13, 1, 1, 1, 1, 1, 1]

        beta_norm_x1.append(beta_vector_x1/beta_x_b1_nominal)
        beta_norm_y1.append(beta_vector_y1/beta_y_b1_nominal)
        beta_norm_x2.append(beta_vector_x2/beta_x_b2_nominal)
        beta_norm_y2.append(beta_vector_y2/beta_y_b2_nominal)

        return beta_norm_x1, beta_norm_y1, beta_norm_x2, beta_norm_y2

    #def define_input():

    #def define_output():

    def get_betabeating(self,test_inputs,estimator):
        # takes and input vector and computes the average beta-beating (IP1 only)
        prediction_test = estimator.predict(test_inputs.reshape(1,-1))
        pred_beta_x1 = prediction_test[0][0:8]
        pred_beta_y1 = prediction_test[0][8:16]
        pred_beta_x2 = prediction_test[0][16:24]
        pred_beta_y2 = prediction_test[0][24:32]

        norm_pred_beta_x1,norm_pred_beta_y1,norm_pred_beta_x2,norm_pred_beta_y2 = self.norm_beta(pred_beta_x1,pred_beta_y1,pred_beta_x2,pred_beta_y2)

        beating_X_B1 = np.mean(abs(norm_pred_beta_x1[0][0:2]))
        beating_Y_B1 = np.mean(abs(norm_pred_beta_y1[0][0:2]))
        beating_B1 = np.mean([beating_X_B1,beating_Y_B1])
        beating_X_B2 = np.mean(abs(norm_pred_beta_x2[0][0:2]))
        beating_Y_B2 = np.mean(abs(norm_pred_beta_y2[0][0:2]))
        beating_B2 = np.mean([beating_X_B2,beating_Y_B2])

        return np.mean([beating_B1,beating_B2])*100
