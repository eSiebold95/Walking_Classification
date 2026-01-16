import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
import XGBmodel
import cnn_model
from time import time

class ModelTrainer:
    """
    Loads data from stored csv files using pandas.
    Returns train and test data as df for fft and np.array for raw.
    """
    
    def __init__(self, random_state: int=42, n_cross_validation:int = 5, data_type: list = ['fft'], rnds_hp_tuning: int = 10, quick_mode: bool = False):
        '''
        Param train_data: 'lab' for the use of lab data only or 'both' for the use of both lab and field data for training.
        Param test_split: float between 0 and 1 indicating the proportion of data to be used for testing.
        Param random_state: random seed for reproducibility.
        Param n_cross_validation: number of cross validation folds.
        '''
        self.root_dir = "/home/elias/2025/sshfs_mounter_2025/data_elias/ECSS_2026/"
        
        # apply self parameters
        self.random_state = random_state
        self.n_cross_validation = n_cross_validation
        self.data_type = data_type
        self.rnds_hp_tuning = rnds_hp_tuning
        self.quick_mode = quick_mode
        
        # list for participant numbers that have been used and those who have not for cross validation
        self.unused_lab_participants = []
        self.unused_field_participants = []
        
        self.used_lab_participants = []
        self.used_field_participants = []
        
        # set random generator
        self.rng = np.random.default_rng(self.random_state)
        
        # dfs for storing results
        self.results_xgb = pd.DataFrame()
        self.results_xgb['metrics'] = ['model', 'n_CV', 'train_acc', 'train_f1', 'test_lab_acc', 'test_lab_f1', 'test_field_acc', 'test_field_f1', 'lr', 'n_estimators', 'max_depth', 'subsample', 'colsample_bytree', 'gamma', 'val_f1']
        
        self.results_cnn = pd.DataFrame()
        self.results_cnn['metrics'] = ['model', 'n_CV', 'train_acc', 'train_f1', 'test_lab_acc', 'test_lab_f1', 'test_field_acc', 'test_field_f1', 
                                       'lr', 'weight_decay', 'batch_size', 'conv1_out_channels', 'conv2_out_channels', 'kernel_size_1', 'kernel_size_2', 'stride_1', 'stride_2', 
                                       'pool_kernel_size_1', 'pool_kernel_size_2', 'dropout', 'batch_norm', 'val_f1']

    def load_csvs(self):
        """
        Loads all data with given type from csv files in the specified directory.
        """
        
        participants_lab = os.listdir(os.path.join(self.root_dir, "fft_lab"))
        participants_field = os.listdir(os.path.join(self.root_dir, "fft_field"))
        
        # shuffle participants
        self.rng.shuffle(participants_lab)
        self.rng.shuffle(participants_field)
        
        # check if there are any empty csv files and remove them
        for part in participants_lab:
            if pd.read_csv(os.path.join(self.root_dir, f"fft_lab", part)).empty:
                participants_lab.remove(part)
        for part in participants_field:
            if pd.read_csv(os.path.join(self.root_dir, f"fft_field", part)).empty:
                participants_field.remove(part)
        
        if 'fft' in self.data_type:
            # load fft lab data
            self.fft_lab_data = []
            for i, part in enumerate(participants_lab):
                if self.quick_mode and i >= 5:
                    break
                # read and change label column to binary
                df = pd.read_csv(os.path.join(self.root_dir, f"fft_lab", part))
                df['label'] = df['label'].map({'no locomotion': 0, 'locomotion': 1})
                df['participant'] = [i]*len(df)
                self.fft_lab_data.append(df)
            
            # set unused participants for cross validation
            self.unused_lab_participants = np.arange(len(self.fft_lab_data)).tolist()
            
            # load fft field data
            self.fft_field_data = []
            for i, part in enumerate(participants_field):
                if self.quick_mode and i >= 5:
                    break
                df = pd.read_csv(os.path.join(self.root_dir, f"fft_field", part))
                df['label'] = df['label'].map({'no locomotion': 0, 'locomotion': 1})
                df['participant'] = [i+100]*len(df)
                self.fft_field_data.append(df)
        
            # set unused participants for cross validation
            self.unused_field_participants = np.arange(len(self.fft_field_data)).tolist()
            
            print(f"Loaded {len(self.fft_lab_data)} lab fft participants and {len(self.fft_field_data)} field fft participants.")
            
        if 'raw' in self.data_type:     
            # load raw lab data
            self.raw_lab_data = []
            for i, part in enumerate(participants_lab):
                if self.quick_mode and i >= 5:
                    break
                df = pd.read_csv(os.path.join(self.root_dir, f"raw_lab", part))
                df['label'] = df['label'].map({'no locomotion': 0, 'locomotion': 1})
                df['participant'] = [i]*len(df)
                self.raw_lab_data.append(df)
                
            # load raw field data
            self.raw_field_data = []
            for i, part in enumerate(participants_field):
                if self.quick_mode and i >= 5:
                    break
                df = pd.read_csv(os.path.join(self.root_dir, f"raw_field", part))
                df['label'] = df['label'].map({'no locomotion': 0, 'locomotion': 1})
                df['participant'] = [i+100]*len(df)
                self.raw_field_data.append(df)
                
            if 'fft' not in self.data_type:
                # set unused participants for cross validation
                self.unused_lab_participants = np.arange(len(self.raw_lab_data)).tolist()
                self.unused_field_participants = np.arange(len(self.raw_field_data)).tolist()
                
            print(f"Loaded {len(self.raw_lab_data)} lab raw participants and {len(self.raw_field_data)} field raw participants.")
            
    def _train_test_split(self):
        """
        Splits the loaded data into train and test sets.
        """
        
        # randomly choose participants for cross validation and keep track of used participants
        loc_cross_val_lab = []
        while (len(loc_cross_val_lab) < len(self.fft_lab_data)//self.n_cross_validation):
            rand_index = self.rng.choice(self.unused_lab_participants)
            loc_cross_val_lab.append(rand_index)
            self.unused_lab_participants.remove(rand_index)
            self.used_lab_participants.append(rand_index)
        
        # same for field data
        loc_cross_val_field = []
        while (len(loc_cross_val_field) < len(self.fft_field_data)//self.n_cross_validation):
            rand_index = self.rng.choice(self.unused_field_participants)
            loc_cross_val_field.append(rand_index)
            self.unused_field_participants.remove(rand_index)
            self.used_field_participants.append(rand_index)
        
        if 'fft' in self.data_type:
            # get ftt train and test data
            self.lab_fft_train = pd.concat([self.fft_lab_data[i] for i in range(len(self.fft_lab_data)) if i not in loc_cross_val_lab], ignore_index=True)
            self.field_fft_train = pd.concat([self.fft_field_data[i] for i in range(len(self.fft_field_data)) if i not in loc_cross_val_field], ignore_index=True)
            self.lab_fft_test = pd.concat([self.fft_lab_data[i] for i in loc_cross_val_lab], ignore_index=True)
            self.field_fft_test = pd.concat([self.fft_field_data[i] for i in loc_cross_val_field], ignore_index=True)
        
        if 'raw' in self.data_type:
            # get raw train and test data
            self.lab_raw_train = pd.concat([self.raw_lab_data[i] for i in range(len(self.raw_lab_data)) if i not in loc_cross_val_lab], ignore_index=True)
            self.field_raw_train = pd.concat([self.raw_field_data[i] for i in range(len(self.raw_field_data)) if i not in loc_cross_val_field], ignore_index=True)
            self.lab_raw_test = pd.concat([self.raw_lab_data[i] for i in loc_cross_val_lab], ignore_index=True)
            self.field_raw_test = pd.concat([self.raw_field_data[i] for i in loc_cross_val_field], ignore_index=True)
    
    def _normalize(self, train_data: pd.DataFrame, test_data: list):
        '''
        Normalize the data using z-score normalization.
        '''
        
        # drop participant column if exists
        if 'participant' in train_data.columns:
            train_data = train_data.drop(columns=['participant'])
        for i in range(len(test_data)):
            if 'participant' in test_data[i].columns:
                test_data[i] = test_data[i].drop(columns=['participant'])
        
        # calculate mean and std from train data
        data_mean = np.array(train_data.drop(columns=['label'])).mean()
        data_std = np.array(train_data.drop(columns=['label'])).std()
        
        # normalize train and test data
        normalized_data = train_data.copy()
        normalized_data.loc[:, normalized_data.columns != 'label'] = (train_data.drop(columns=['label']) - data_mean) / data_std
        
        # normalize test data
        if len (test_data) == 1:
            normalized_test_data = test_data[0].copy()
            normalized_test_data.loc[:, normalized_test_data.columns != 'label'] = (test_data[0].drop(columns=['label']) - data_mean) / data_std
            return normalized_data, normalized_test_data
        
        else:
            normalized_test_data1 = test_data[0].copy()
            normalized_test_data1.loc[:, normalized_test_data1.columns != 'label'] = (test_data[0].drop(columns=['label']) - data_mean) / data_std
            normalized_test_data2 = test_data[1].copy()
            normalized_test_data2.loc[:, normalized_test_data2.columns != 'label'] = (test_data[1].drop(columns=['label']) - data_mean) / data_std
            return normalized_data, normalized_test_data1, normalized_test_data2
    
    def _balance_data(self, lab_train_data: pd.DataFrame, field_train_data: pd.DataFrame = None, seed: int = 42):
        """
        Balances training data to have equal number of samples for each class.
        """
        
        # get min samples per class
        lab_data_non_loc = lab_train_data[lab_train_data['label'] == 0]
        lab_data_loc = lab_train_data[lab_train_data['label'] == 1]
        min_samples = min(len(lab_data_non_loc), len(lab_data_loc))
        
        # if field data is provided, consider it for balancing as well
        if field_train_data is not None:
            field_data_non_loc = field_train_data[field_train_data['label'] == 0]
            field_data_loc = field_train_data[field_train_data['label'] == 1]
            min_samples = min(min_samples, len(field_data_non_loc), len(field_data_loc))
        
        # concatenate balanced lab data
        train_data = pd.concat([lab_data_non_loc.sample(n=min_samples, random_state=seed),
                                       lab_data_loc.sample(n=min_samples, random_state=seed)],
                                      ignore_index=True)
        
        # if field data is provided, concatenate balanced field data
        if field_train_data is not None:
            field_data_balanced = pd.concat([field_data_non_loc.sample(n=min_samples, random_state=seed),
                                            field_data_loc.sample(n=min_samples, random_state=seed)],
                                           ignore_index=True)
            train_data = pd.concat([train_data, field_data_balanced], ignore_index=True)
        
        # shuffle the final training data
        train_data = shuffle(train_data, random_state=seed)
        return train_data
    
    def sample_hyperparameters(self,):
        '''
        samples hyperparameters for XGB model using random uniform distribution
        samples hyperparameters for CNN model using random uniform distribution
        '''
        
        ### valid range for xgb hyperparameters
        xgb_round_hp = {
            'lr': float(self.rng.uniform(0.01, 0.2)),
            'n_estimators': int(self.rng.integers(10, 50)),
            'max_depth': int(self.rng.integers(3, 11)),
            'subsample': float(self.rng.uniform(0.5, 1.0)),
            'colsample_bytree': float(self.rng.uniform(0.5, 1.0)),
            'gamma': float(10 ** self.rng.uniform(-4, 0.7))
            }
        
        ### valid rage for cnn hyperparameters
        cnn_round_hp = {
            # optimizer / training
            'lr': float(10 ** self.rng.uniform(-4.5, -2.5)),
            'weight_decay': float(10 ** self.rng.uniform(-6, -3)),            
            'batch_size': int(self.rng.choice([16, 32, 64, 128])),

            # architecture
            'conv1_out_channels': int(self.rng.choice([4, 8, 16])),
            'conv2_out_channels': int(self.rng.choice([8, 16, 32])),
            'kernel_size_1': int(self.rng.choice([3, 5, 7, 9])),
            'kernel_size_2': int(self.rng.choice([3, 5, 7])),
            'stride_1': int(self.rng.choice([1, 2, 3])),
            'stride_2': int(self.rng.choice([1, 2, 3])),
            'pool_kernel_size_1': int(self.rng.choice([2, 3])),
            'pool_kernel_size_2': int(self.rng.choice([2, 3])),

            # regularization
            'dropout': float(self.rng.uniform(0.0, 0.5)),
            'batch_norm': bool(self.rng.choice([True, False]))
        }

        return xgb_round_hp, cnn_round_hp
    
    def hyperparameter_search(self, data_fft, data_raw):
        """
        Placeholder for hyperparameter search method.
        Searches hyperparameters for XGB model using random search and holdout validation. (50 iterations)
        Searches hyperparameters for CNN model using random search and holdout validation. (50 iterations)
        Searches are always based on the same distribution for XGB and CNN.
        1/3 of data is used for validation in each iteration.
        """
        print("Starting hyperparameter search...")
        # dataframe to store fft results
        xgb_hp_dict = {
            'lr': [],
            'n_estimators': [],
            'max_depth': [],
            'subsample': [],
            'colsample_bytree': [],
            'gamma': [],
            'val_f1': []
        }
        
        # dataframe to store cnn results
        cnn_hp_dict = {
            'lr': [],
            'weight_decay': [],
            'batch_size': [],
            'conv1_out_channels': [],
            'conv2_out_channels': [],
            'kernel_size_1': [],
            'kernel_size_2': [],
            'stride_1': [],
            'stride_2': [],
            'pool_kernel_size_1': [],
            'pool_kernel_size_2': [],
            'dropout': [],
            'batch_norm': [],
            'val_f1': []
        }
        
        # loop for cross-validation
        for i in range(self.rnds_hp_tuning):
            # holdout validation set
            holdout_participants =data_fft['participant'].nunique() // 3
            val_participants = self.rng.choice(data_fft['participant'].unique(), size=holdout_participants, replace=False)
            
            # fft train and val split
            val_fft =data_fft[data_fft['participant'].isin(val_participants)].drop(columns=['participant'])
            train_fft =data_fft[~data_fft['participant'].isin(val_participants)].drop(columns=['participant'])
            val_fft.reset_index(drop=True, inplace=True)
            train_fft.reset_index(drop=True, inplace=True)

            # raw train and val split
            val_raw =data_raw[data_raw['participant'].isin(val_participants)].drop(columns=['participant'])
            train_raw =data_raw[~data_raw['participant'].isin(val_participants)].drop(columns=['participant'])
            val_raw.reset_index(drop=True, inplace=True)
            train_raw.reset_index(drop=True, inplace=True)
            
            # sample hyperparameters
            xgb_round_hp, cnn_round_hp = self.sample_hyperparameters()
            
            '''xgb model for fft data'''
            # create xgb model
            model = XGBmodel.XGBModel(model_name=None, n_CV=0, hyperparameters=xgb_round_hp, train_data=train_fft, test_lab_data=val_fft, test_field_data=None)
            # train model
            model.train()
            # evaluate model
            f1_score = model.evaluate()
            
            # add to results df
            for key in xgb_round_hp.keys():
                xgb_hp_dict[key].append(xgb_round_hp[key])
            xgb_hp_dict['val_f1'].append(f1_score)
            
            '''cnn model for raw data'''
            # normalize data
            train_raw, val_raw = self._normalize(train_raw,  [val_raw])
            
            # create cnn model
            model = cnn_model.CNN_Model(model_name = None, hyperparameters=cnn_round_hp)
            # train model
            model.train_model(train_raw)
            # evaluate model
            f1_score, _ = model.evaluate_set(val_raw)
            
            # add to results df
            for key in cnn_round_hp.keys():
                cnn_hp_dict[key].append(cnn_round_hp[key])
            cnn_hp_dict['val_f1'].append(f1_score)
            
            if (i+1) % 5 == 0:
                print(f"{i+1}/{self.rnds_hp_tuning} iterations done: mean XGB F1: {np.mean(xgb_hp_dict['val_f1']):.3f}, mean CNN F1: {np.mean(cnn_hp_dict['val_f1']):.3f} (time elapsed: {((time() - self.start_time)//60):.0f} minutes {(time() - self.start_time)%60:.2f} seconds)")
                        
        # get the best hyperparameters for xgb
        xgb_param_df = pd.DataFrame(xgb_hp_dict)
        best_xgb_param = xgb_param_df.loc[xgb_param_df['val_f1'].idxmax()].to_dict()
        
        # get the best hyperparameters for cnn
        cnn_param_df = pd.DataFrame(cnn_hp_dict)
        best_cnn_param = cnn_param_df.loc[cnn_param_df['val_f1'].idxmax()].to_dict()
        
        print(f"Hyperparameter search done. (time elapsed: {((time() - self.start_time)//60):.0f} minutes {(time() - self.start_time)%60:.2f} seconds)")
        print('-'*50)
        
        # return best hyperparameters
        return best_xgb_param, best_cnn_param
            
        
    def training_loop(self):
        """
        Selects training data based on 
        """
        print("-"*50)
        
        # start time measurement
        self.start_time = time()
        
        # loop for cross-validation
        for i in range(self.n_cross_validation):
            # split data into train and test sets
            self._train_test_split()
            
            ''' hyperparameter search  for XGB and CNN for LAB ONLY'''
            xgb_param, cnn_param = self.hyperparameter_search(self.lab_fft_train, self.lab_raw_train)
            
            '''XGB LAB ONLY'''
            # balance data
            fft_train = self._balance_data(lab_train_data = self.lab_fft_train)
            # create model
            model_name = f'XGB_fft_lab_CV{i+1}'
            model = XGBmodel.XGBModel(model_name=model_name, n_CV=i+1, hyperparameters = xgb_param, train_data=fft_train, test_lab_data=self.lab_fft_test, test_field_data=self.field_fft_test)
            # train model
            model.train()
            # evaluate model
            results = model.evaluate()
            # add to results df
            self.results_xgb = pd.concat([self.results_xgb, results], ignore_index=True, axis =1)
            print(f"completed cross-validation fold {i+1}/{self.n_cross_validation} of {model_name}")
            print(f"Train Accuracy: {results.loc[2, f'{model_name}_{i+1}']:.4f}, Test Lab Accuracy: {results.loc[4, f'{model_name}_{i+1}']:.4f} Test Field Accuracy: {results.loc[6, f'{model_name}_{i+1}']:.4f}, (time elapsed: {((time() - self.start_time)//60):.0f} minutes {(time() - self.start_time)%60:.2f} seconds)")
            print("-"*50)
            
            '''CNN LAB ONLY'''
            # normalize data
            raw_train, raw_test_lab, raw_test_field = self._normalize(self.lab_raw_train, [self.lab_raw_test, self.field_raw_test])
            # balance data
            raw_train = self._balance_data(lab_train_data = raw_train)
            # create model
            model_name = f'CNN_raw_lab_CV{i+1}'
            model = cnn_model.CNN_Model(model_name=model_name, hyperparameters=cnn_param)
            # train model
            model.train_model(raw_train)
            # evaluate model
            results = model.evaluate_model(raw_train, raw_test_lab, raw_test_field)
            # add to results df
            self.results_cnn = pd.concat([self.results_cnn, results], ignore_index=True, axis =1)
            print(f"completed cross-validation fold {i+1}/{self.n_cross_validation} of {model_name}")
            print(f"Train Accuracy: {results.loc[2, f'{model_name}']:.4f}, Test Lab Accuracy: {results.loc[4, f'{model_name}']:.4f}, Test Field Accuracy: {results.loc[6, f'{model_name}']:.4f}, (time elapsed: {((time() - self.start_time)//60):.0f} minutes {(time() - self.start_time)%60:.2f} seconds)")
            print("-"*50)
            
            '''XGB LAB + FIELD'''
            # balance data for bot XGB and CNN
            combined_train_fft = self._balance_data(lab_train_data = self.lab_fft_train, field_train_data = self.field_fft_train)
            combined_train_raw = self._balance_data(lab_train_data = self.lab_raw_train, field_train_data = self.field_raw_train)   
            # run hyperparameter search
            xgb_param, cnn_param = self.hyperparameter_search(combined_train_fft, combined_train_raw)
            # create model
            model_name = f'XGB_fft_lab_field_CV{i+1}'
            model = XGBmodel.XGBModel(model_name=model_name, n_CV=i+1, hyperparameters=xgb_param, train_data=combined_train_fft, test_lab_data=self.lab_fft_test, test_field_data=self.field_fft_test)
            # train model
            model.train()
            # evaluate model
            results = model.evaluate()
            print(f"Completed cross-validation fold {i+1}/{self.n_cross_validation} of {model_name}")
            print(f"Train Accuracy: {results.loc[2, f'{model_name}_{i+1}']:.4f}, Test Lab Accuracy: {results.loc[4, f'{model_name}_{i+1}']:.4f}, Test Field Accuracy: {results.loc[6, f'{model_name}_{i+1}']:.4f}, (time elapsed: {((time() - self.start_time)//60):.0f} minutes {(time() - self.start_time)%60:.2f} seconds)")
            print("-"*50)
            # add to results df
            self.results_xgb = pd.concat([self.results_xgb, results], ignore_index=True, axis = 1)
            
            '''CNN LAB + FIELD''' 
            # normalize data
            combined_train_raw, raw_test_lab, raw_test_field = self._normalize(combined_train_raw, [self.lab_raw_test, self.field_raw_test])
            # create model
            model_name = f'CNN_raw_lab_field_CV{i+1}'
            model = cnn_model.CNN_Model(model_name=model_name, hyperparameters=cnn_param)
            # train model
            model.train_model(combined_train_raw)
            # evaluate model
            results = model.evaluate_model(combined_train_raw, raw_test_lab, raw_test_field)
            print(f"Completed cross-validation fold {i+1}/{self.n_cross_validation} of {model_name}")
            print(f"Train Accuracy: {results.loc[2, f'{model_name}']:.4f}, Test Lab Accuracy: {results.loc[4, f'{model_name}']:.4f}, Test Field Accuracy: {results.loc[6, f'{model_name}']:.4f} (time elapsed: {((time() - self.start_time)//60):.0f} minutes {(time() - self.start_time)%60:.2f} seconds)")
            print("-"*50)
            # add to results df
            self.results_cnn = pd.concat([self.results_cnn, results], ignore_index=True, axis = 1)
            
    def save_results(self, file_name: str = 'results'):
        """
        Saves the results dataframe to a csv file.
        """
        
        # save xgb results
        self.results_xgb.columns = self.results_xgb.iloc[0, :]
        self.results_xgb = self.results_xgb.drop(index=0).reset_index(drop=True)
        self.results_xgb.to_csv(f"{file_name}_xgb.csv", index=False)
        print(f"Results saved to {file_name}_xgb.csv")
        print(self.results_xgb)
        
        # save cnn results
        self.results_cnn.columns = self.results_cnn.iloc[0, :]
        self.results_cnn = self.results_cnn.drop(index=0).reset_index(drop=True)
        self.results_cnn.to_csv(f"{file_name}_cnn.csv", index=False)
        print(f"Results saved to {file_name}_cnn.csv")
        print(self.results_cnn)
