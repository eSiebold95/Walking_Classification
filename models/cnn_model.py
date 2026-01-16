import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd

class CNN(nn.Module):
    def __init__(self, input_size, hyperparameters):
        super().__init__()

        self.hyperparameters = hyperparameters
        self.relu = nn.ReLU()

        c1 = int(hyperparameters['conv1_out_channels'])
        c2 = int(hyperparameters['conv2_out_channels'])

        self.conv1 = nn.Conv1d(input_size[0], c1, kernel_size=int(hyperparameters['kernel_size_1']), stride=int(hyperparameters['stride_1']), padding=0)
        self.maxpool1 = nn.MaxPool1d(kernel_size=int(hyperparameters['pool_kernel_size_1']), stride=2, padding=0)
        self.batchnorm1 = nn.BatchNorm1d(c1)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=int(hyperparameters['kernel_size_2']), stride=int(hyperparameters['stride_2']), padding=0)
        self.maxpool2 = nn.MaxPool1d(kernel_size=int(hyperparameters['pool_kernel_size_2']), stride=2, padding=0)
        self.batchnorm2 = nn.BatchNorm1d(c2)
        self.dropout = nn.Dropout(p=float(hyperparameters['dropout']))

        # infer features
        with torch.no_grad():
            x = torch.zeros(1, input_size[0], input_size[1])
            x = self.maxpool1(self.relu(self.conv1(x)))
            x = self.maxpool2(self.relu(self.conv2(x)))
            n_features = x.flatten(1).shape[1]

        self.flatten = nn.Flatten()
        self.output_binary = nn.Linear(n_features, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        if self.hyperparameters['batch_norm']:
            x = self.batchnorm1(x)
        x = self.maxpool1(x)

        x = self.relu(self.conv2(x))
        if self.hyperparameters['batch_norm']:
            x = self.batchnorm2(x)
        x = self.maxpool2(x)

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.output_binary(x)
        return x  # logits
    
# data loader
class DataLoaderAcc(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
 
class CNN_Model:
    '''
    Creates a CNN model for binary classification using PyTorch.
    Args:
        input_size (tuple): A tuple representing the size of the input data (channels, length).
        hyperparameters (dict): A dictionary containing hyperparameters for the CNN model.
    '''
    def __init__(self, model_name, hyperparameters, input_size: tuple = (1, 260)):
        # hyperparameters and model name
        self.model_name = model_name
        self.hyperparameters = hyperparameters
        
        # define model
        self.model = CNN(input_size=input_size, hyperparameters=hyperparameters)

        # define criteria
        self.criterion = nn.BCEWithLogitsLoss()

        # define optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'])
        
    def train_model(self, train_data: pd.DataFrame):
        '''
        Reshapes train data to proper shape
        Trains model on 25 epochs
        '''
        # train data reshape
        self.train = train_data
        self.X = np.array(self.train.drop(columns=['label']).values).reshape(-1, 1, 260).astype(np.float32)
        self.y = np.array(self.train['label'].values).reshape(-1, 1).astype(np.float32)
        
        # data loader
        train_loader = DataLoader(DataLoaderAcc(self.X, self.y), batch_size=int(self.hyperparameters['batch_size']), shuffle=True)
        
        # training loop
        epochs = 25
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                # zero the parameter gradients
                self.optimizer.zero_grad() # reset gradients
                
                # forward pass
                outputs = self.model(inputs) # forward propagation
                
                # compute loss
                loss = self.criterion(outputs, labels) # loss calculation
                
                # backward pass and optimization
                loss.backward() # backpropagation
                self.optimizer.step() # update weights
                
    def evaluate_model(self, train_data: pd.DataFrame, test_lab_data: pd.DataFrame, test_field_data: pd.DataFrame):
        '''
        Evaluates model on lab and field test data
        '''
        # list to store results
        results = []
        
        # append model name and n_CV
        results.append(self.model_name)
        results.append(int(self.model_name.split('CV')[-1]))
        
        for data in [train_data, test_lab_data, test_field_data]:
            f1, accuracy = self.evaluate_set(data)
            
            # append results
            results.append(accuracy)
            results.append(f1)
            
        # append hyperparameters
        for key in self.hyperparameters.keys():
            results.append(self.hyperparameters[key])
            
        # turn to pd dataframe and return
        results_series = pd.DataFrame({
            f'{self.model_name}': results
        })
        
        return results_series
        
                
    def evaluate_set(self, test_data: pd.DataFrame):
        '''
        Reshapes test data to proper shape
        Evaluates model on test data
        '''
        # test data reshape
        self.test = test_data
        self.X_test = np.array(self.test.drop(columns=['label']).values).reshape(-1, 1, 260).astype(np.float32)
        self.y_test = np.array(self.test['label'].values).reshape(-1,).astype(np.float32)
        
        # turn into tensors
        self.X_test_tensor = torch.tensor(self.X_test)
        
        # evaluation loop
        self.model.eval() # set model to evaluation mode
        with torch.no_grad():
            # predict
            outputs = self.model(self.X_test_tensor)
            predicted = (torch.sigmoid(outputs) >= 0.5).float()
            
        # calculate metrics
        f1 = f1_score(self.y_test, predicted.numpy().reshape(-1,))
        accuracy = accuracy_score(self.y_test, predicted.numpy().reshape(-1,))
        
        return f1, accuracy
            