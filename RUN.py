import sys
import os.path
import RLLHC
import pandas as pd
from time import time
import numpy as np
import matplotlib.cm as cm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MaxAbsScaler
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
                
                if len(mqt_errors_b1[0])==0:
                    pass
                else:
                    mqt_errors_b1, mqt_errors_b2 = self.rllhc.clean_data(mqt_errors_b1, mqt_errors_b2)
                    errors_tmp = np.concatenate(( \
                    np.vstack(triplet_errors),\
                    np.vstack(mqt_errors_b1), np.vstack(mqt_errors_b2), \
                    np.vstack(arc_errors_b1), np.vstack(arc_errors_b2), \
                    ), axis=1)

                    betas_tmp = np.concatenate((
                    np.vstack(delta_beta_star_x_b1), \
                    np.vstack(delta_beta_star_y_b1), \
                    np.vstack(delta_beta_star_x_b2), \
                    np.vstack(delta_beta_star_y_b2), \
                    np.vstack(delta_muy_b1), np.vstack(delta_muy_b2), \
                    np.vstack(delta_mux_b1), np.vstack(delta_mux_b2), \
                    np.vstack(n_disp_b1), np.vstack(n_disp_b2), \
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

            if model == 'Ridge':

                print('Running Ridge+Bagging')
                self.bagging = False
                self.estimator = self.rllhc.mymodel(self.train_inputs,self.train_outputs,self.bagging)
                #print(self.estimator)
                self.rllhc.get_model_score(self.estimator,self.train_inputs,self.train_outputs,self.test_inputs,self.test_outputs)
        
                print(self.estimator.get_params())

                cross_valid = True
                if cross_valid:
                    # Some cross-validation results
                    score = self.rllhc.get_cross_validation(self.train_inputs,self.train_outputs)
                    print("Cross validation: score = %1.4f +/- %1.4f" % (np.mean(score), np.std(score)))

                features = False
                if features:
                    self.rllhc.get_feature_importance(self.estimator,mode)

                modelfile = 'Ridge_surrogate_20k.pkl'
                pickle.dump(self.estimator,open(modelfile, 'wb'))

            if model == 'RandomForest':

                print('Running RandomForest -- estimators = {}, depth = {}'.format(n_estimators, depth))

                n_estimators = 50
                depth = 5
                min_samples_split = 2
                min_samples_leaf = 1

                self.estimator = self.rllhc.RFmodel(n_estimators, depth, min_samples_leaf, min_samples_split, self.train_inputs, self.train_outputs)
                self.rllhc.get_model_score(self.estimator,self.train_inputs,self.train_outputs,self.test_inputs,self.test_outputs)

                print(self.estimator.get_params())
      
                # Some cross-validation results
                score = self.rllhc.get_cross_validation(self.train_inputs,self.train_outputs)
                print(score)
                print("Cross validation: score = %1.2f +/- %1.2f" % (np.mean(score), np.std(score)))

                features = False
                if features:
                    # get importance
                    importance = self.estimator.feature_importances_
                    for i,v in enumerate(importance):
                        print('Feature: %0d, Score: %.5f' % (i,v))
                    plt.bar([x for x in range(len(importance))], importance)
                    plt.show()

                gridsearch = False
                if gridsearch:

                    param_grid = {
                    'bootstrap': [True],
                    'max_depth': [20],
                    'max_features': ['auto'],
                    'min_samples_leaf': [1],
                    'min_samples_split': [2,3],
                    'n_estimators': [70]
                    }

                    gridsearch = GridSearchCV(self.estimator, param_grid=param_grid, cv=5, verbose=1)
                    gridsearch.fit(self.train_inputs, self.train_outputs)

                    # Best cross validation score
                    print('Cross Validation Score:', gridsearch.best_score_)
                    # Best parameters which resulted in the best score
                    print('Best Parameters:', gridsearch.best_params_)


        if mode == 'predictor':
            # in this mode inputs and outputs are swapped
            print("Predictor model: input = betas, output = errors")

            X = self.betas
            y = self.errors

            scale = False
            if scale:
                print('Scaled fetures')
                transformer = MaxAbsScaler().fit(X)
                X = transformer.transform(X)

            # Split data in train and test
            self.train_inputs, self.test_inputs, self.train_outputs, self.test_outputs = train_test_split(X, y, test_size=0.2, random_state=None) 

            print("Number of train samples = {}".format(len(self.train_inputs)))
            print("Number of test samples = {}".format(len(self.test_inputs)))
            print("Number of input features {}".format(len(self.train_inputs[0])))
            print("Number of output features = {}".format(len(self.test_outputs[0])))

            print('Running Ridge+Bagging')
            self.bagging = False
            estimator = self.rllhc.mymodel(self.train_inputs,self.train_outputs,self.bagging)
            self.rllhc.get_model_score(estimator,self.train_inputs,self.train_outputs,self.test_inputs,self.test_outputs)

            cross_valid = True
            if cross_valid:
                # Some cross-validation results
                score = self.rllhc.get_cross_validation(self.train_inputs,self.train_outputs)
                print("Cross validation: score = %1.4f +/- %1.4f" % (np.mean(score), np.std(score)))

            features = False
            if features:
                self.rllhc.get_feature_importance(self.estimator,mode)

            doGridSearch = True
            if doGridSearch:
                
                param_grid = {
                    'alpha': [2e-4,3e-4],
                    'copy_X': [True],
                    'fit_intercept': [True],
                    'max_iter': [None],
                    'normalize': [True, False],
                    'random_state': [0], 
                    'solver': ['auto'],
                    'tol': [1e-50]
                }

                gridsearch = GridSearchCV(estimator, param_grid=param_grid, cv=5, verbose=1)
                gridsearch.fit(self.train_inputs, self.train_outputs)

                # Best cross validation score
                print('Cross Validation Score:', gridsearch.best_score_)

                # Best parameters which resulted in the best score
                print('Best Model:', gridsearch.best_estimator_)

                estimator = gridsearch.best_estimator_
                self.rllhc.get_model_score(estimator,self.train_inputs,self.train_outputs,self.test_inputs,self.test_outputs)

            case = True
            if case:
                predict_case = estimator.predict(self.test_inputs[0].reshape(1,-1))
                print(np.shape(predict_case))
                plt.plot(predict_case.reshape(-1,1), label='prediction')
                plt.plot(self.test_outputs[0], label='real')
                plt.legend()
                output_label = [
                    'Q3BL', 'Q3AL', 'Q2BL', 'Q2AL', 'Q1BL', 'Q1AL',
                    'Q1AR', 'Q1BR', 'Q2AR', 'Q2BR', 'Q3AR', 'Q3BR',
                    'MQT_1', 'MQT_2', 'MQT_3', 'MQT_4'
                    ]
                #plt.xticks(np.arange(0.5,16.5,1), output_label, rotation=0)
                plt.show()

            modelfile = 'Ridge_predictor_80k.pkl'
            pickle.dump(estimator,open(modelfile, 'wb'))
        
       
if __name__ == "__main__":
    if len(sys.argv) > 3:
        path_to_data = sys.argv[1]
        num_samples = int(sys.argv[2])
        mode = sys.argv[3]
        model = sys.argv[4]
        
    if len(sys.argv) <= 3:
        print()
        print("          ERROR: One or more arguments are missing: ")
        print("          How to Run: % RUN.py <path_to_data> <num_samples> <mode: surrogate, predictor> <model: Ridge, RandomForest>")
        print()
        exit()
    
    t0 = time()
    f = Run(path_to_data,num_samples)
    f.run(mode)
    print("Time required = %1.2f seconds" % float(time() - t0))