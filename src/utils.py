import os
import sys
import pickle
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import custom_exception
from sklearn.metrics import r2_score

def save_object(filePath,obj):
    try:
        dir_path= os.path.dirname(filePath)

        os.makedirs(dir_path,exist_ok=True)

        with open(filePath, 'wb') as file_obj:
            pickle.dump(obj,file_obj)
    
    except Exception as e:
        logging.info('Error in dumping Pickle File:')
        custom_exception(e,sys)
        
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            # Predictions
            y_pred = model.predict(X_test)

            # Evaluate the model performance
            test_model_score = r2_score(y_test, y_pred)
            report[model_name] = test_model_score
        return report
    except Exception as e:
        logging.info('Error in Evaluating the model:')
        custom_exception(e, sys)


def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise custom_exception(e,sys)