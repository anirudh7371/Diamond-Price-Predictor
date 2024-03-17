import os 
import pandas as pd
import sys
from src.logger import logging
from src.exception import custom_exception
from src.components.data_ingestion import dataIngestion
from src.components.data_transformation import dataTransformation
from src.components.model_trainer import modelTrainer

if __name__=='__main__':
    obj=dataIngestion()
    trainDataPath,testDataPath= obj.initiateDataIngestion()
    print(trainDataPath,testDataPath)

    data_transformation=dataTransformation()
    train_arr,test_arr,_=data_transformation.initiateDataTransformation(trainDataPath,testDataPath)

    model_train=modelTrainer()
    model_train.initiateModelTrainer(train_arr,test_arr)