import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    model_trainer_obj_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:

            #Splitting train and test obtained from data_transformer into X and Y counterparts  
            logging.info("Splitting train and test data into X and Y counterparts")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            # Different types of models
            models = {
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor":KNeighborsRegressor(),
                "XGBClassifier":XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor(),
            }


            # Calling utils.py file for evaluating model. which returns{ 'model name':'score' }
            model_report:dict=evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)


            # Getting the Best model score 
            best_model_score = max(sorted(model_report.values()))

            # Getting the best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            # Getting the best model
            best_model = models[best_model_name]

            if(best_model_score < 0.6):
                raise CustomException("No best model found")

            logging.info(f"Best found model on both training and testing dataset")
            logging.info(f'Best model: {best_model}')

            # Creating a pickle file for the best_model
            save_object(
                file_path=self.model_trainer_config.model_trainer_obj_path,
                obj=best_model
            )

            '''
            Using the best model to again predict
            '''
            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)
