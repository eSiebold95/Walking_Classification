import pandas as pd
import numpy as np
import os
from datetime import datetime as dt, timedelta as td
from numpy import fft


class LabClassifier:
    def __init__(self, root_dir: str, output_raw: str, output_fft: str, participant_id: str, overlapping_samples:int = 130):
        ### set output path and participant id and gyroscope
        self.output_raw = output_raw
        self.output_fft = output_fft
        self.participant_id = participant_id
        
        ### set parameters for reshaping
        self.window = 260
        self.overlap = overlapping_samples
        self.step = self.window - self.overlap
        
        # read data
        self.data = pd.read_csv(f"{root_dir}/{participant_id}")
        
        print(f"data read for {participant_id}...", end=' ')
        
        ### reshape data
        self.reshape_data()
        print("data reshaped...", end=' ')
        
        ### check labels
        self.check_labels()
        print("labels checked...", end=' ')
        
        ### perform fft
        self.perform_fft()
        print("fft performed...", end=' ')
        
        ### save prepared data
        self.save_prepared_data()
        print("data saved.")
        
    def reshape_data(self):
        '''
        Reshape data into a n x 260 array to prepare for fft processing.
        based on window size, overlap and step size.
        '''        
        mag_acc = self.data['mag_acc'].to_numpy()
        labels = self.data['model_label_acc'].to_numpy()
        
        # get the sliding windows with certain overlap
        self.mag_acc = np.lib.stride_tricks.sliding_window_view(mag_acc, self.window)[::self.step]
        self.labels = np.lib.stride_tricks.sliding_window_view(labels, self.window)[::self.step]
        
    def check_labels(self):
        '''
        Check homogenity of labels in each window and
        perform extra checks:
        - set window to unknown if label = resting but std of mag is high (>70)
        - set window to unknown if label = active but std of mag is low (<70) 
        '''
        
        # set homogeneous labels to single value, heterogeneous to unknown (0)
        labels = self.labels[:, 0] * (np.ptp(self.labels, axis = 1) == 0).astype(int)
        
        # remove all unknown labels
        mask = (labels > 0)
        self.labels = pd.Series(labels[mask], name = 'label')
        self.mag_acc = self.mag_acc[mask]
        
        # map labels back to resting (1), active (2), walking (3), running (4)
        def label_map(x):
            if x == 1 or x == 2:
                return 'no locomotion'
            elif x == 3 or x == 4:
                return 'locomotion'
            
        self.labels = self.labels.map(label_map)
        
        # to dataframe
        self.merged_data = pd.DataFrame(self.mag_acc)
        self.merged_data.columns = [f'acc_{i+1}' for i in range(self.window)]
        self.merged_data['label'] = self.labels.reset_index(drop=True)
        self.merged_data.reset_index(drop=True, inplace=True)
        
        # save
        self.merged_data.to_csv(f"{self.output_raw}/{self.participant_id}", index=False)
        
    def perform_fft(self):
        '''
        Perform FFT on reshaped data.
        '''
        
        # set parameters
        N = 260
        T = 1/52
        
        # apply hamming window
        window = np.hamming(N)
        self.windowed_acc = self.mag_acc * window.reshape(1, -1)
        
        # fft transformation
        self.fft_acc = fft.fft(self.windowed_acc)

        #frequencies
        self.frequencies = fft.fftfreq(N, T)

        #positive magnitudes
        self.pos_fft_acc = np.abs(self.fft_acc)[:, :N//2]
        
        # fft in dataframe format
        self.fft_acc_df = pd.DataFrame(self.pos_fft_acc)
        
        # set column titles
        self.fft_acc_df.columns = [f'acc_{i:.1f}' for i in np.arange(0, np.max(self.frequencies) + 0.2, 0.2)]
        
        # get only frequencies between 1.0 and 10.0 Hz
        self.fft_acc_df = self.fft_acc_df[[f'acc_{i:.1f}' for i in np.arange(1, 10.2, 0.2)]].copy()
        
    def save_prepared_data(self):
        '''concatenates fft data and labels and saves to csv'''
        self.prepared_data = pd.concat([self.fft_acc_df, self.labels], axis = 1)
        self.prepared_data.to_csv(f"{self.output_fft}/{self.participant_id}", index=False)
        
        
    
        
        
    
        

            
