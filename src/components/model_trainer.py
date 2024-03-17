import pandas as pd
import numpy as np
import os
import sys
from src.logger import logging
from src.exception import custom_exception
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,AdaBoostRegressor
from src.utils import evaluate_model
from src.utils import save_object

#Model Training Configuration

@dataclass
class modelTrainerConfig:
    modelTrainerFilePath=os.path.join('artifacts','model.pkl')

#Model training
class modelTrainer:
    def __init__(self):
        self.model_trainer_config= modelTrainerConfig()

    def initiateModelTrainer(self,train_array,test_array):
        try:
            models={
                'linear regression': LinearRegression(),
                'ridge':Ridge(),
                'lasso':Lasso(),
                'elasticnet':ElasticNet(),
                'knn':KNeighborsRegressor(),
                'gradientboosting':GradientBoostingRegressor(),
                'random forest':RandomForestRegressor(),
                'adaboost':AdaBoostRegressor()
            }

            logging.info('Splitting the Dataset into training Dataset and test Dataset')
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n============================================')
            logging.info(f'Model Report: {model_report}')

            #To get best model score from dictionary
            best_model_score=max(sorted(model_report.values()))
            best_model_name= list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            print(f'Best Model Found, Model Name:{best_model_name} , R2_Score:{best_model_score}')
            print('\n')
            print('='*20)
            logging.info(f'Best Model Found, Model Name:{best_model_name} , R2_Score:{best_model_score}')

            save_object(
                filePath=self.model_trainer_config.modelTrainerFilePath,
                obj=best_model
            )
        except Exception as e:
            logging.info('Exception Occured as Model Training')
            raise custom_exception(e,sys)
