import os
import sys
import dill

from src.exception import CustomException

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


# Makes a pickle file for preprocessor and is called from data_transformation.py
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


# Called from Model Trainer
# Evaluate the model 
def evaluate_model(X_train, y_train, X_test, y_test, models, params):

    try:
        
        # Declaring a dictionary
        report = {}

        for i in range(len(list(models))):

            # Accessing the values of dictionary
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]


            # Goes over all the possible combination and selects the best possible parameters
            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train, y_train)

            '''
             Although this would be suffice, we need to use model because the model here is also used over at model trainer
             So, despite gs.fit() being completely fine. I have used model.set_params(**gs.best_params_)
            '''
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Adding key and value to dictionary(report){'model name':'score'}
            report[list(models.keys())[i]] = test_model_score

            return report
        
    except Exception as e:
        raise CustomException(e, sys)
    
# Loading the pkl files
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)