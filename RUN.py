import sys
import os.path
import RLLHC
import pandas as pd
from time import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pickle

class Run: 
     
    def __init__(self, path_to_data, num_samples):

        self.rllhc = RLLHC.RLLHC()
    
        for i in range(num_samples):
            
            run = 'run'+str(i+1).zfill(4)

            self.filename = path_to_data + run +'/training_set.npy'
            if os.path.isfile(self.filename):
                self.all_samples = np.load('{}'.format(self.filename), allow_pickle=True, encoding='latin1')

                delta_beta_star_x_b1, delta_beta_star_y_b1, delta_beta_star_x_b2, \
                delta_beta_star_y_b2, delta_mux_b1, delta_muy_b1, delta_mux_b2, \
                delta_muy_b2, n_disp_b1, n_disp_b2, \
                triplet_errors, arc_errors_b1, arc_errors_b2, mqt_errors_b1, mqt_errors_b2 = self.all_samples.T

                mqt_errors_b1, mqt_errors_b2 = self.rllhc.clean_data(mqt_errors_b1, mqt_errors_b2)

                errors_tmp = np.concatenate(( \
                np.vstack(triplet_errors),\
                #np.vstack(arc_errors_b1), np.vstack(arc_errors_b2), \
                #np.vstack(mqt_errors_b1), np.vstack(mqt_errors_b2),
                ), axis=1)

                betas_tmp = np.concatenate((np.vstack(delta_beta_star_x_b1), np.vstack(delta_beta_star_y_b1), \
                np.vstack(delta_beta_star_x_b2), np.vstack(delta_beta_star_y_b2), \
                #np.vstack(delta_mux_b1), np.vstack(delta_mux_b2), \
                #np.vstack(delta_muy_b1), np.vstack(delta_muy_b2), \
                #np.vstack(n_disp_b1), np.vstack(n_disp_b2), \
                ), axis=1)

                if i == 0:
                    self.betas = betas_tmp
                    self.errors = errors_tmp
                else:
                    self.betas = np.concatenate((self.betas, betas_tmp, ), axis=0)
                    self.errors = np.concatenate((self.errors, errors_tmp, ), axis=0)
            else:
                print ("File does not exist")

    def run(self,mode):

        if mode == 'surrogate':

            print()
            print("Surrogate model: input = errors, output = betas")
            
            self.input_data = self.errors
            self.output_data = self.betas

            # Split data in train and test
            self.train_inputs, self.test_inputs, self.train_outputs, self.test_outputs = train_test_split(
            self.input_data, self.output_data, test_size=0.2, random_state=None) 

            print("Number of train samples = {}".format(len(self.train_inputs)))
            print("Number of test samples = {}".format(len(self.test_inputs)))
            print("Number of input features {}".format(len(self.train_inputs[0])))
            print("Number of output features = {}".format(len(self.test_outputs[0])))
            
            n_estimators = 50
            depth = 5
            min_samples_split = 2
            min_samples_leaf = 1

            print('Running RandomForest -- estimators = {}, depth = {}'.format(n_estimators, depth))

            self.estimator = self.rllhc.RFmodel(n_estimators, depth, min_samples_leaf, min_samples_split, self.train_inputs, self.train_outputs)
            self.rllhc.get_model_score(self.estimator,self.train_inputs,self.train_outputs,self.test_inputs,self.test_outputs)

            #print(self.estimator.get_params())
      
            # Some cross-validation results
            score = self.rllhc.get_cross_validation(self.train_inputs,self.train_outputs)
            print(score)
            print("Cross validation: score = %1.2f +/- %1.2f" % (np.mean(score), np.std(score)))

            # get importance
            #importance = self.estimator.feature_importances_
            # summarize feature importance
            #for i,v in enumerate(importance):
            #    print('Feature: %0d, Score: %.5f' % (i,v))
            # plot feature importance
            #plt.bar([x for x in range(len(importance))], importance)
            #plt.show()

            print("Average beta-beating = ", self.rllhc.get_betabeating(self.test_inputs[0],self.estimator))

            # hyperparameter optimization
            param_grid = {
            'bootstrap': [True],
            'max_depth': [20],
            'max_features': ['auto'],
            'min_samples_leaf': [1],
            'min_samples_split': [2,3],
            'n_estimators': [70]
            }

            #gridsearch = GridSearchCV(self.estimator, param_grid=param_grid, cv=5, verbose=1)

            #gridsearch.fit(self.train_inputs, self.train_outputs)

            # Best cross validation score
            #print('Cross Validation Score:', gridsearch.best_score_)

            # Best parameters which resulted in the best score
            #print('Best Parameters:', gridsearch.best_params_)

        if mode == 'predictor':
            # in this mode inputs and outputs are swapped
            print("Predictor model: input = betas, output = errors")

            self.input_data = self.betas
            self.output_data = self.errors

            # Split data in train and test
            self.train_inputs, self.test_inputs, self.train_outputs, self.test_outputs = train_test_split(
            self.input_data, self.output_data, test_size=0.2, random_state=None) 

            print("Number of train samples = {}".format(len(self.train_inputs)))
            print("Number of test samples = {}".format(len(self.test_inputs)))
            print("Number of input features {}".format(len(self.train_inputs[0])))
            print("Number of output features = {}".format(len(self.test_outputs[0])))

            print('Running Ridge+Bagging')
            self.bagging = False
            self.estimator = self.rllhc.mymodel(self.train_outputs,self.train_inputs,self.bagging)
            self.rllhc.get_model_score(self.estimator,self.train_outputs,self.train_inputs,self.test_outputs,self.test_inputs)
        
            print(self.estimator.get_params())

            # Some cross-validation results
            #score = self.rllhc.get_cross_validation(self.train_outputs,self.train_inputs)
            #print("Cross validation: score = %1.2f +/- %1.2f" % (np.mean(score), np.std(score)))

            # hyperparameter optimization
            param_grid = {
                'alpha': [0.001, 0.005, 0.01],
                #'copy_X': [True],
                #'fit_intercept': [True],
                #'max_iter': [None],
                'normalize': [True, False],
                #'random_state': [0], 
                'solver': ['auto'],
                'tol': [1e-50]
            }

            gridsearch = GridSearchCV(self.estimator, param_grid=param_grid, cv=5, verbose=1)
            gridsearch.fit(self.train_outputs, self.train_inputs)

            # Best cross validation score
            print('Cross Validation Score:', gridsearch.best_score_)

            # Best parameters which resulted in the best score
            print('Best Parameters:', gridsearch.best_params_)

        
        if mode == 'xgb':
            # in this mode inputs and outputs are swapped
            print("Predictor model: input = betas, output = errors")
            print('Running XGBooster Regression')
            self.estimator = self.rllhc.myXGBoost(self.train_outputs,self.train_inputs)
            self.rllhc.get_model_score(self.estimator,self.train_outputs,self.train_inputs,self.test_outputs,self.test_inputs)
       
if __name__ == "__main__":
    if len(sys.argv) > 3:
        path_to_data = sys.argv[1]
        num_samples = int(sys.argv[2])
        mode = sys.argv[3]
        
    if len(sys.argv) <= 3:
        print()
        print("          ERROR: One or more arguments are missing: ")
        print("          How to Run: % RUN.py <path_to_data> <num_samples> <mode: surrogate, predictor> ")
        print()
        exit()
    
    t0 = time()
    f = Run(path_to_data,num_samples)
    f.run(mode)
    print("Time required = %1.2f seconds" % float(time() - t0))