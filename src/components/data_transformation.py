import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


    # Defining the pipelines
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            num_features = ['reading_score', 'writing_score']
            cat_features = [
                'gender', 
                'race_ethnicity', 
                'parental_level_of_education', 
                'lunch',
                'test_preparation_course'
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder())
                ]
            )

            logging.info("Numerical colums standard scaling completed")
            logging.info(f"Numerical colums: {num_features}")
            logging.info("Categorical colums encoding completed")
            logging.info(f"Categorical colums: {cat_features}")

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", num_pipeline, num_features),
                    ("categorical_pipeline", cat_pipeline, cat_features)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        

    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading test and train data completed")
            logging.info("Obtaining preprocessing object")

            # Preprocess from previous function
            preprocessing_obj = self.get_data_transformer_object()

            
            target_column_name="math_score"

            # Removing target values from the dataset(train.csv)
            target_feature_train_df=train_df[target_column_name]
            input_feature_train_df=train_df.drop(columns=[target_column_name], axis=1)

            # Removing target values from the dataset(test.csv)
            target_feature_test_df=test_df[target_column_name]
            input_feature_test_df=test_df.drop(columns=[target_column_name], axis=1)

            logging.info("Applying preprocessing object on training and testing dataframe")


            # Applying the preprocessor on the datasets
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)


            # Concatinating the processed input feature with target column
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")



            # Calls utils.py file to make a pickle file for the preprocessot
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )


            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)