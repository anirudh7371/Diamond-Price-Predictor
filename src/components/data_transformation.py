import pandas as pd
import numpy as np
import os
import sys
from src.logger import logging
from src.exception import custom_exception
from dataclasses import dataclass

from sklearn.impute import SimpleImputer #Handling Missing Values
from sklearn.preprocessing import StandardScaler,OrdinalEncoder #Handling Categorical values and normalizing the data
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object

#Data Transformation Configuration

@dataclass
class dataTranformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')

#Data Transformation
class dataTransformation:
    def __init__(self):
        self.data_transform_config=dataTranformationConfig()

    def getDataTransformationObj(self):

        try:
            logging.info('Data Transformation Initiated')
            #Segragating numerical and categorical columns
            numerical_columns= ['carat','depth','table','x','y','z']
            categorical_columns=['cut','color','clarity']


            #Defining the custom ranking for each ordinal 
            cut_categories= ['Fair','Good','Very Good','Premium','Ideal']
            clarity_categories=['IF','VVS1','VVS2','VS1','VS2','SI1','SI2','I1']
            color_categories=['D','E','F','G', 'H', 'I','J']

            #Numerical pipeline

            logging.info('Numerical Pipeline Initiated')

            num_pipeline=Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
                ]
            )

            #Categorical Pipeline

            logging.info('Numerical Pipeline Initiated')

            cat_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scaler' ,StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer(
            transformers=[
                ('numerical_pipeline',num_pipeline,numerical_columns),
                ('categorical_pipeline',cat_pipeline,categorical_columns)
                ]
            )
            logging.info("Pipeline Completed")

            return preprocessor
            
        
        except Exception as e:
            logging.info('Error in Data Transformation:')
            raise custom_exception(e,sys)

    def initiateDataTransformation(self,train_path,test_path):
        try:
            #Reading training and Test Path
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Reading of Train and Test data Done')
            logging.info(f'Train Dataframe Head:\n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head:\n{test_df.head().to_string()}')

            logging.info('Obtaining Processing Data')

            preprocessor_obj= self.getDataTransformationObj()

            targetColumn='price'
            dropColumn= [targetColumn,'id']

            #Independent and Dependent Columns
            inputFeatureTrainDf= train_df.drop(columns=dropColumn,axis=1)
            targetFeatureTrainDf=train_df[targetColumn]

            inputFeatureTestDf= test_df.drop(columns=dropColumn,axis=1)
            targetFeatureTestDf=test_df[targetColumn]

            #Transformation

            input_feature_train_arr=preprocessor_obj.fit_transform(inputFeatureTrainDf)
            input_feature_test_arr=preprocessor_obj.transform(inputFeatureTestDf)

            logging.info("Applying Preprocessing object on training and test set.")

            trainArray=np.c_[input_feature_train_arr,np.array(targetFeatureTrainDf)]
            testArray=np.c_[input_feature_test_arr,np.array(targetFeatureTestDf)]

            save_object(
                filePath=self.data_transform_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info('Preprocessing Done!')

            return (
                trainArray,
                testArray,
                self.data_transform_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            logging.info('Error in Data Transformation Initialization:')
            custom_exception(e,sys)

