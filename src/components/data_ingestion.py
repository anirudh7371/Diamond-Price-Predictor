import pandas as pd
import os
import sys
from src.logger import logging
from src.exception import custom_exception
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

#Initializing the Data Ingestion Configuration
@dataclass
class dataIngestionConfig:
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    raw_data_path = os.path.join('artifacts','raw.csv')

#Data Ingestion Method
class dataIngestion:
    def __init__(self):
        self.ingestionConfig=dataIngestionConfig()

    def initiateDataIngestion(self):
        logging.info('Data Ingestion Method Starts!')

        try:
            
            df=pd.read_csv(os.path.join('notebooks/data','gemstone.csv'))
            logging.info("Dataset Read as Panda's Dataframe")

            os.makedirs(os.path.dirname(self.ingestionConfig.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestionConfig.raw_data_path,index=False)

            #Train Test Split
            logging.info('Train Test Split')
            trainSet,testSet=train_test_split(df,test_size=0.3,random_state=42)
            trainSet.to_csv(self.ingestionConfig.train_data_path,index=False,header=True)
            testSet.to_csv(self.ingestionConfig.test_data_path,index=False,header=True)

            return(
                self.ingestionConfig.test_data_path,
                self.ingestionConfig.train_data_path
            )

        except Exception as e:
            logging.info("Error occured in Data Ingestion",e)