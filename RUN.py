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
from sklearn.metrics import mean_absolute_error
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

    def run(self,mode,model):

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

                cross_valid = False
                if cross_valid:
                    # Some cross-validation results
                    score = self.rllhc.get_cross_validation(self.train_inputs,self.train_outputs)
                    print("Cross validation: score = %1.4f +/- %1.4f" % (np.mean(score), np.std(score)))

                features = False
                if features:
                    self.rllhc.get_feature_importance(self.estimator,mode)

                modelfile = 'Ridge_surrogate_20k.pkl'
                pickle.dump(self.estimator,open(modelfile, 'wb'))

                prediction_test = self.estimator.predict(self.test_inputs)

                plt.figure(figsize=(10,5))
                plt.plot(prediction_test[0:100,0], alpha=0.7, label='Prediction')
                plt.plot(self.test_outputs[0:100,0], alpha=0.7, label='True')
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.ylabel(r'$\Delta\beta$ [m]', fontsize=18)
                plt.xlabel('seed #', fontsize=18)
                plt.legend(fontsize=18)
                plt.savefig('pred_beta.pdf',bbox_inches='tight')
                plt.show()

            if model == 'RandomForest':

                n_estimators = 50
                depth = 5
                min_samples_split = 2
                min_samples_leaf = 1

                print('Running RandomForest -- estimators = {}, depth = {}'.format(n_estimators, depth))

                self.estimator = self.rllhc.RFmodel(n_estimators, depth, min_samples_leaf, min_samples_split, self.train_inputs, self.train_outputs)
                self.rllhc.get_model_score(self.estimator,self.train_inputs,self.train_outputs,self.test_inputs,self.test_outputs)

                print(self.estimator.get_params())
      
                # Some cross-validation results
                score = self.rllhc.get_cross_validation(self.train_inputs,self.train_outputs)
                print(score)
                print("Cross validation: score = %1.2f +/- %1.2f" % (np.mean(score), np.std(score)))

                features = True
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

            if model == 'Ridge':
                print('Running Ridge')
                self.bagging = False
                estimator = self.rllhc.mymodel(self.train_inputs,self.train_outputs,self.bagging)
                self.rllhc.get_model_score(estimator,self.train_inputs,self.train_outputs,self.test_inputs,self.test_outputs)

            if model == 'RandomForest':

                n_estimators = 10
                depth = 5
                min_samples_split = 2
                min_samples_leaf = 1

                print('Running RandomForest -- estimators = {}, depth = {}'.format(n_estimators, depth))

                estimator = self.rllhc.RFmodel(n_estimators, depth, min_samples_leaf, min_samples_split, self.train_inputs, self.train_outputs)
                self.rllhc.get_model_score(estimator,self.train_inputs,self.train_outputs,self.test_inputs,self.test_outputs)

                print(estimator.get_params())

            cross_valid = False
            if cross_valid:
                # Some cross-validation results
                score = self.rllhc.get_cross_validation(self.train_inputs,self.train_outputs)
                print("Cross validation: score = %1.4f +/- %1.4f" % (np.mean(score), np.std(score)))

            features = True
            if features:
                self.rllhc.get_feature_importance(estimator,mode)

            doGridSearch = False
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

            case = False
            if case:
                predict_case = estimator.predict(self.test_inputs[0].reshape(1,-1))
                print(np.shape(predict_case))
                plt.plot(predict_case.reshape(-1,1)*1e4, label='prediction')
                plt.plot(self.test_outputs[0]*1e4, label='real')
                plt.legend()
                output_label = [
                    'Q3BL', 'Q3AL', 'Q2BL', 'Q2AL', 'Q1BL', 'Q1AL',
                    'Q1AR', 'Q1BR', 'Q2AR', 'Q2BR', 'Q3AR', 'Q3BR',
                    'MQT_1', 'MQT_2', 'MQT_3', 'MQT_4'
                    ]
                #plt.xticks(np.arange(0.5,16.5,1), output_label, rotation=0)
                plt.xlabel('Magnet number')
                plt.ylabel('Error [10^{-4}]')
                plt.show()

                # predict systematic error after averaging over all triplet quads
                # select quads

            systematic = True
            # Enables detailed study of systematic error in the triplet

            if systematic:
                prediction_test = estimator.predict(self.test_inputs)

                KQX3B1 = -0.0234346323
                KQX3B2 = 0.0234346323
                KQX2B1 = 0.04005580581
                KQX2B2 = -0.04005580581
                KQX1B1 = -0.02385208
                KQX1B2 = 0.02385208
                
                KQX = [KQX3B1, KQX3B1, KQX2B2, KQX2B2,
                        KQX1B1, KQX1B1, KQX1B2, KQX1B2,
                        KQX2B1, KQX2B1, KQX3B2, KQX3B2]

                pred_rel = prediction_test[:,0:12]/KQX*1e4
                real_rel = self.test_outputs[:,0:12]/KQX*1e4

                residual_rel =  real_rel - pred_rel

                Q3BL_pred_rel = pred_rel[:,0]
                Q3AL_pred_rel = pred_rel[:,1]
                Q2BL_pred_rel = pred_rel[:,2]
                Q2AL_pred_rel = pred_rel[:,3]
                Q1BL_pred_rel = pred_rel[:,4]
                Q1AL_pred_rel = pred_rel[:,5]

                Q1AR_pred_rel = pred_rel[:,6]
                Q1BR_pred_rel = pred_rel[:,7]
                Q2AR_pred_rel = pred_rel[:,8]
                Q2BR_pred_rel = pred_rel[:,9]
                Q3AR_pred_rel = pred_rel[:,10]
                Q3BR_pred_rel = pred_rel[:,11]

                Q3BL_real_rel = real_rel[:,0]
                Q3AL_real_rel = real_rel[:,1]
                Q2BL_real_rel = real_rel[:,2]
                Q2AL_real_rel = real_rel[:,3]
                Q1BL_real_rel = real_rel[:,4]
                Q1AL_real_rel = real_rel[:,5]

                Q1AR_real_rel = real_rel[:,6]
                Q1BR_real_rel = real_rel[:,7] 
                Q2AR_real_rel = real_rel[:,8]
                Q2BR_real_rel = real_rel[:,9] 
                Q3AR_real_rel = real_rel[:,10]
                Q3BR_real_rel = real_rel[:,11]

                plt.figure()
                plt.plot(Q2AL_real_rel,Q2AL_pred_rel-Q2AL_real_rel,'.',label='Q2AL')
                plt.plot(Q2AR_real_rel,Q2AR_pred_rel-Q2AR_real_rel,'.',label='Q2AR')
                plt.plot(Q2BL_real_rel,Q2BL_pred_rel-Q2BL_real_rel,'.',label='Q2BL')
                plt.plot(Q2BR_real_rel,Q2BR_pred_rel-Q2BR_real_rel,'.',label='Q2BR')
                plt.xlabel('True relative error [$10^{-4}$]',fontsize=18)
                plt.ylabel('True - Prediction [$10^{-4}$]',fontsize=18)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.legend(fontsize=18)
                plt.savefig('Q2_error_scatter_plot.pdf',bbox_inches='tight')


                MAE = mean_absolute_error(real_rel,pred_rel)
                MAE_Q1AL = mean_absolute_error(Q1AL_real_rel,Q1AL_pred_rel)
                MAE_Q1BL = mean_absolute_error(Q1BL_real_rel,Q1BL_pred_rel)
                MAE_Q1AR = mean_absolute_error(Q1AR_real_rel,Q1AR_pred_rel)
                MAE_Q1BR = mean_absolute_error(Q1BR_real_rel,Q1BR_pred_rel)
                MAE_Q2AL = mean_absolute_error(Q2AL_real_rel,Q2AL_pred_rel)
                MAE_Q2BL = mean_absolute_error(Q2BL_real_rel,Q2BL_pred_rel)
                MAE_Q2AR = mean_absolute_error(Q2AR_real_rel,Q2AR_pred_rel)
                MAE_Q2BR = mean_absolute_error(Q2BR_real_rel,Q2BR_pred_rel)
                MAE_Q3AL = mean_absolute_error(Q3AL_real_rel,Q3AL_pred_rel)
                MAE_Q3BL = mean_absolute_error(Q3BL_real_rel,Q3BL_pred_rel)
                MAE_Q3AR = mean_absolute_error(Q3AR_real_rel,Q3AR_pred_rel)
                MAE_Q3BR = mean_absolute_error(Q3BR_real_rel,Q3BR_pred_rel)

                print('MAE = %1.2f' % MAE)
                print('MAE(Q1AL) = %1.2f' % MAE_Q1AL)
                print('MAE(Q1BL) = %1.2f' % MAE_Q1BL) 
                print('MAE(Q1AR) = %1.2f' % MAE_Q1AR) 
                print('MAE(Q1BR) = %1.2f' % MAE_Q1BR)
                print('MAE(Q2AL) = %1.2f' % MAE_Q2AL)
                print('MAE(Q2BL) = %1.2f' % MAE_Q2BL) 
                print('MAE(Q2AR) = %1.2f' % MAE_Q2AR) 
                print('MAE(Q2BR) = %1.2f' % MAE_Q2BR) 
                print('MAE(Q3AL) = %1.2f' % MAE_Q3AL)
                print('MAE(Q3BL) = %1.2f' % MAE_Q3BL) 
                print('MAE(Q3AR) = %1.2f' % MAE_Q3AR) 
                print('MAE(Q3BR) = %1.2f' % MAE_Q3BR) 

                syst_err_pred = []
                syst_err_real = []
                for i in range(len(pred_rel)):
                        syst_err_pred.append(np.mean(pred_rel[i,:]))
                        syst_err_real.append(np.mean(real_rel[i,:]))

                MAE_syst = mean_absolute_error(syst_err_pred,syst_err_real)
                print('MAE Systematic error %1.2f' % MAE_syst)

                plt.figure()
                plt.hist(syst_err_pred,bins=20,range=(-15,15),alpha=0.5,label='Predicted')
                plt.hist(syst_err_real,bins=20,range=(-15,15),alpha=0.5,label='True')
                plt.xlabel('$\Delta K/K$ syst. [$10^{-4}$]',fontsize=18)
                plt.xticks(fontsize=18)
                plt.ylabel('Counts',fontsize=18)
                plt.yticks(fontsize=18)
                plt.legend(fontsize=18)
                plt.savefig('syst_error_hist.pdf',bbox_inches='tight')

                plt.figure()
                plt.plot(syst_err_pred,label='Predicted')
                plt.plot(syst_err_real,label='True')
                plt.ylabel('$\Delta K/K$ syst. [$10^{-4}$]',fontsize=18)
                plt.xticks(fontsize=18)
                plt.xlabel('Seed number',fontsize=18)
                plt.yticks(fontsize=18)
                plt.legend(fontsize=18)
                plt.savefig('syst_error.pdf',bbox_inches='tight')

                plt.figure()
                plt.hist(Q2BR_pred_rel,bins=20,range=(-30,30),alpha=0.5,label='Predicted')
                plt.hist(Q2BR_real_rel,bins=20,range=(-30,30),alpha=0.5,label='True')
                plt.xlabel('$\Delta K/K$ [$10^{-4}$]',fontsize=18)
                plt.xticks(fontsize=18)
                plt.ylabel('Counts',fontsize=18)
                plt.yticks(fontsize=18)
                plt.legend(fontsize=18)
                plt.savefig('error_hist.pdf',bbox_inches='tight')

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
    f.run(mode,model)
    print("Time required = %1.2f seconds" % float(time() - t0))