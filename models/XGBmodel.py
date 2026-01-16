import os
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

class XGBModel:
    """
    Wrapper class for XGBoost model for classification tasks.
    """
    
    def __init__(self, model_name: str, n_CV: int,  hyperparameters: dict, train_data: pd.DataFrame, test_lab_data: pd.DataFrame, test_field_data: pd.DataFrame = None):
        
        # Initialize param
        self.model_name = model_name
        self.n_CV = n_CV
        self.hyperparameters = hyperparameters
        self.train_data = train_data
        self.test_lab_data = test_lab_data
        self.test_field_data = test_field_data
        
        # check if there is a column 'participant' in any of the dataframes and drop it
        if 'participant' in self.train_data.columns:
            self.train_data = self.train_data.drop(columns=['participant'])
        if 'participant' in self.test_lab_data.columns:
            self.test_lab_data = self.test_lab_data.drop(columns=['participant'])
        if self.test_field_data is not None and 'participant' in self.test_field_data.columns:
            self.test_field_data = self.test_field_data.drop(columns=['participant'])
        
    def train(self):
        """
        Trains the XGBoost model on the provided training data.
        """
        self.X_train = self.train_data.drop(columns=['label'])
        self.y_train = self.train_data['label']
        
        self.model = XGBClassifier(n_estimators = int(self.hyperparameters['n_estimators']), learning_rate = self.hyperparameters['lr'], max_depth = int(self.hyperparameters['max_depth']), subsample = self.hyperparameters['subsample'], 
                                   colsample_bytree = self.hyperparameters['colsample_bytree'], gamma = self.hyperparameters['gamma'], eval_metric='logloss')
        self.model.fit(self.X_train, self.y_train)
        
        
    def evaluate(self):
        """
        Evaluates the trained model on the provided train and test data.
        returns a pandas Series with accuracy and F1-score for train, test_lab, and test_field datasets.
        """ 
        # train prediction
        train_pred = self.model.predict(self.X_train)
        train_acc = accuracy_score(self.y_train, train_pred)
        train_f1 = f1_score(self.y_train, train_pred, average='weighted')
        
        # test prediction lab
        test_lab_pred = self.model.predict(self.test_lab_data.drop(columns=['label']))
        test_lab_acc = accuracy_score(self.test_lab_data['label'], test_lab_pred)
        test_lab_f1 = f1_score(self.test_lab_data['label'], test_lab_pred, average='weighted')
        
        # test prediction field
        if self.test_field_data is not None:
            test_field_pred = self.model.predict(self.test_field_data.drop(columns=['label']))
            test_field_acc = accuracy_score(self.test_field_data['label'], test_field_pred)
            test_field_f1 = f1_score(self.test_field_data['label'], test_field_pred, average='weighted')
            
            # pd series
            df = pd.DataFrame({
                f'{self.model_name}_{self.n_CV}': [self.model_name, self.n_CV, train_acc, train_f1, test_lab_acc, test_lab_f1, test_field_acc, test_field_f1] + list(self.hyperparameters.values())})
            return df
        else:
            # for validation with validation set only
            return test_lab_f1
            
            
        
    