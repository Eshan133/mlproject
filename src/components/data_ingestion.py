import os
import sys

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass



# Making directory(artifacts) to store raw, train and test dataset
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")



class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Reading data from the given dataset(can be CSV file, or any other database)
            # Change it as per your need
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info("Read the dataset as dataframe")


            # Making directories with the data_path that was defined
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok= True)


            # Storing the data(df) to raw data file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)


            # Splitting the df(raw data) into train and test
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df, test_size=0.2, random_state=42)
            

             # Storing the data(train_set) to train data file
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
             # Storing the data(test_set) to test data file
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is complete")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

'''
To execute the ingestion file run(python src/components/data_ingestion.py) command
'''
if __name__=="__main__":
    obj=DataIngestion()
    train_data, test_data=obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_array, test_array, _= data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_array, test_array))

'''
Steps for data ingestion:
a. Read the dataset
b. Store the dataset into a file
c. Split the data into train and test dataset
d. Store train and test into their respective files
e. return the path for train.csv and test.csv
--------------------------------------------
Output
Creates a artifact folder with data.csv, train.csv and test.csvv file
'''