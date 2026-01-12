import numpy as np
from numpy import fft
from scipy.signal import butter, filtfilt
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from datetime import datetime, timedelta
import os
import time

'''
time adjustment needs to be set manually if there is one - in init
'''

class FieldClassifier:
    '''
    Classifies field data based on GPS speed into walking and not walking using predefined thresholds.
    functions "classify", "filter_data", "transpose_window", "merge_data", and "apply_fft" to process and classify the data.
    Classification is 'not walking' and 'walking'
    Outputs raw snips (260 sample windows) with classification as well as FFT magnitudes with classification. 
    '''
    
    def __init__ (self, root_dir_acc: str, root_dir_gps: str, participant_id:str, output_path_fft: str, output_path_raw: str, hour_offset: int = 0):
        # set output path and participant id
        self.output_path_fft = output_path_fft
        self.output_path_raw = output_path_raw
        if len(participant_id) > 5:
            self.id = participant_id[:2] + participant_id[3:]
        else:
            self.id = participant_id
        
        # initialization message
        start_time = time.time()
        print(f"Initializing FieldClassifier for {self.id}...", end=' ')
           
        ### filter settings
        sampling_rate = 52
        nyquist = sampling_rate / 2
        low = 0.5 / nyquist
        order = 4
        self.b, self.a = butter(order, low, btype='high')
        
        ### set thresholds for extra classification
        '''
        Definition of extra windows
        last / next 3 windows > 3.5 km/h from gps
        current speed < 6 and > 3.5 km/h
        walking > 0.1 & active > 0.5 
        '''
        self.walking_speed_min = 3.5  # km/h
        self.walking_speed_max = 10  # km/h
        self.resting_speed_max = 1  # km/h
        #self.walking_threshold = 0.1  # prob
        #self.active_threshold = 0.5  # prob
        
        ### read acc data
        df_acc_l = []
        for dirs in os.listdir(os.path.join(root_dir_acc, participant_id)):
            df_acc_l.append(pd.read_parquet(os.path.join(root_dir_acc, participant_id, dirs, 'acc.parquet')))
        df_acc = pd.concat(df_acc_l, axis=0, ignore_index=False)
        df_acc.sort_index(inplace=True)
        
        # set time offset
        self.time_offset = timedelta(hours = hour_offset)
        
         # time and magnitude for acceleration data
        df_acc['time'] = pd.to_datetime(df_acc.index) + self.time_offset
        df_acc.reset_index(drop=True, inplace=True)
        df_acc['mag'] = (df_acc['X']**2 + df_acc['Y']**2 + df_acc['Z']**2)**0.5
        self.df_acc = df_acc[['time', 'mag']].copy()
        
        # cut for march time
        self.df_acc = self.df_acc[(self.df_acc['time'] >= pd.Timestamp('2025-11-19 13:30:00')) & (self.df_acc['time'] <= pd.Timestamp('2025-11-20 13:00:00'))].copy()
        
        ### read gps data
        self.df_watch = pd.read_csv(os.path.join(root_dir_gps, self.id + '.csv'))
        self.df_watch = self.df_watch[['Time', 'Speed (km/h)']].copy()
        
        # time formatting watch data
        self.df_watch['Time'] = pd.to_datetime(self.df_watch['Time'])
        self.df_watch.reset_index(drop=True, inplace=True)
        self.df_watch.columns = ['time', 'speed']
        
        # cut for march time
        self.df_watch = self.df_watch[(self.df_watch['time'] >= pd.Timestamp('2025-11-19 13:30:00')) & (self.df_watch['time'] <= pd.Timestamp('2025-11-20 13:00:00'))].copy()
        
        # emw forward and backward smoothing of speed data
        emw_forward = self.df_watch['speed'].ewm(span = 5, adjust=False).mean()
        emw_backward = self.df_watch['speed'][::-1].ewm(span = 5, adjust=False).mean()[::-1]
        self.df_watch['speed'] = (emw_forward + emw_backward) / 2
        
        print(f'data read ({time.time() - start_time:.2f} s)...', end=' ')
        
        ### process data
        self.filter_data()
        self.classify()
        print(f'data classified ({time.time() - start_time:.2f} s)...', end=' ')
        self.transpose_window()
        print(f'data windowed ({time.time() - start_time:.2f} s)...', end=' ')
        self.merge_data()
        print(f'data merged ({time.time() - start_time:.2f} s)...', end=' ')
        self.apply_fft()
        print(f'data FFT applied ({time.time() - start_time:.2f} s).')
        
        
    def filter_data(self):
        self.df_acc['mag'] = filtfilt(self.b, self.a, self.df_acc.loc[:, 'mag'])
        
    def classify(self):
        '''
        classify data based on speed thresholds defined in init
        '''
        speed = np.array(self.df_watch['speed'])
        walking_classification = ((np.roll(speed, 5) > self.walking_speed_min) & 
                                  (np.roll(speed, 4) > self.walking_speed_min) & 
                                  (np.roll(speed, 3) > self.walking_speed_min) & 
                                  (np.roll(speed, 2) > self.walking_speed_min) & 
                                  (np.roll(speed, 1) > self.walking_speed_min) & 
                                  (speed > self.walking_speed_min) & (speed < self.walking_speed_max) & 
                                  (np.roll(speed, -1) > self.walking_speed_min) & 
                                  (np.roll(speed, -2) > self.walking_speed_min) & 
                                  (np.roll(speed, -3) > self.walking_speed_min) &
                                  (np.roll(speed, -4) > self.walking_speed_min) &
                                  (np.roll(speed, -5) > self.walking_speed_min)).astype(int)*2
        resting_classification = ((np.roll(speed, 5) < self.resting_speed_max) & 
                                  (np.roll(speed, 4) < self.resting_speed_max) & 
                                  (np.roll(speed, 3) < self.resting_speed_max) & 
                                  (np.roll(speed, 2) < self.resting_speed_max) & 
                                  (np.roll(speed, 1) < self.resting_speed_max) & 
                                  (speed < self.resting_speed_max) & 
                                  (np.roll(speed, -1) < self.resting_speed_max) & 
                                  (np.roll(speed, -2) < self.resting_speed_max) & 
                                  (np.roll(speed, -3) < self.resting_speed_max) & 
                                  (np.roll(speed, -4) < self.resting_speed_max) & 
                                  (np.roll(speed, -5) < self.resting_speed_max)).astype(int)
        
        classification = walking_classification + resting_classification
        self.df_watch['classification'] = classification
        
        
    def transpose_window(self):
        '''
        Transpose acceleration data into windows of 260 samples without overlap
        '''
        window_size = 260
        mags = np.array(self.df_acc['mag'])
        num_windows = len(mags)//window_size
        mags = mags[:num_windows * window_size]
        
        time_series = self.df_acc['time'][window_size//2::window_size][:num_windows]
        mags = mags.reshape((num_windows, window_size))
        self.df_windows = pd.DataFrame(mags)
        self.df_windows.columns = [f'acc_{i+1}' for i in range(window_size)]
        self.df_windows['time'] = time_series.values
        
    def merge_data(self):
        '''
        Merge transposed acceleration data with watch data based on time
        '''
        self.df_merged = pd.merge_asof(self.df_windows.sort_values('time'), 
                                       self.df_watch.sort_values('time'), 
                                       on='time', direction='nearest', tolerance=pd.Timedelta('2s'))
    
        self.df_merged.reset_index(drop=True, inplace=True)
        self.df_merged.dropna(inplace=True)
        self.df_merged.reset_index(drop=True, inplace=True)
        
        # mask based on classification = 0 (not detected)
        self.df_merged = self.df_merged[self.df_merged['classification'] != 0].copy()
        
        # change classification 1 to 0 and 2 to 1
        self.df_merged['label'] = self.df_merged['classification'].replace({1:'no locomotion', 2:'locomotion'})
        
        # drop time and classification column
        self.df_merged.drop(columns=['time', 'classification', 'speed'], inplace=True)
        
        # save merged data
        self.df_merged.to_csv(os.path.join(self.output_path_raw, f'{self.id}.csv'), index=False)
        
    def apply_fft(self):
        '''
        Apply FFT to the transposed windows and return a dataframe with magnitudes
        '''
        
        self.mag_windows = np.array(self.df_merged.loc[:, [f'acc_{i+1}' for i in range(260)]])
        
        # set parameters
        N = 260
        T = 1/52
        
        #apply hamming window
        hamming_window = np.hamming(N).reshape(1, -1)
        windowed_mag = self.mag_windows*hamming_window

        #Fourier Transformation
        fft_windowed_mag = np.fft.fft(windowed_mag, axis = 1)
        
        # frequencies
        frequencies = np.fft.fftfreq(N, T)
        
        #magnitudes
        fft_spectrum = np.abs(fft_windowed_mag[:, :N // 2])
        
        # into dataframe
        df_fft = pd.DataFrame(fft_spectrum)
        df_fft.columns = [f'acc_{i:.1f}' for i in np.arange(0, np.max(frequencies) + 0.2, 0.2)]
        self.df_fft = df_fft[[f'acc_{i:.1f}' for i in np.arange(1, 10.2, 0.2)]].copy()
        
        # fuse with classifications and time
        self.df_fft['label'] = self.df_merged['label'].values
        
        # save fft data
        self.df_fft.to_csv(os.path.join(self.output_path_fft, f'{self.id}.csv'), index=False)
        
        
        
        

        

        
        
        
