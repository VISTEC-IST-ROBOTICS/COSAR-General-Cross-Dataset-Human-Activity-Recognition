# data folder: loader.py
from typing import Any, Literal
from collections import defaultdict
from itertools import product
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch

import plotly.graph_objects as go
from plotly.subplots import make_subplots


class DataLoader:
    DATASET_CONFIGS = {
        'pamap2': {
            'num_subjects':  9,
            'sampling_rate': 100,
            'file_extension': 'dat',
            'dataset_path':  '~/dataset/pamap2/PAMAP2_Dataset/Protocol/subject10',
            'imu_positions': ['hand', 'chest', 'ankle'],
            'column_prefix': ['hand', 'chest', 'ankle'],
            'column_extra':  ['timestamp', 'label', 'heartrate'],
            'column_drop':   ['label', 'timestamp', 'heartrate'],
            'sensor_types':  ['temp', 'accel16', 'accel6', 'gyro', 'mag', 'orient'],
            'activities': {
                'transient': 0, 'lying': 1, 'sitting': 2, 'standing': 3,
                'walking': 4, 'running': 5, 'cycling': 6, 'NordicWalking': 7,
                'watching_TV': 9, 'computer_work': 10, 'car_driving': 11,
                'AcdS': 12, 'DcdS': 13, 'vacumm_cleaning': 16, 'ironing': 17,
                'folding_laundry': 18, 'house_cleaning': 19, 'playing_soccer': 20,
                'rope_jumping': 24,
            },
        },
        'mhealth': {
            'num_subjects':  10,
            'sampling_rate': 50,
            'file_extension': 'log',
            'dataset_path':  f'~/dataset/mhealth/MHEALTHDATASET/mHealth_subject',
            'imu_positions': ['ankle', 'hand'],
            'column_prefix': ['ankle', 'hand'],
            'column_extra':  ['chest_accel_x', 'chest_accel_y', 'chest_accel_z', 'ecg_1', 'ecg_2'],
            'column_drop':   None,
            'sensor_types':  ['accel', 'gyro', 'mag'],
            'activities': {
                'standing': 1, 'sitting': 2, 'lying': 3, 'walking': 4,
                'AcdS': 5, 'waist_bending': 6, 'FEA': 7, 'crouching': 8,
                'cycling': 9, 'jogging': 10, 'running': 11, 'JFB': 12,
            },
        },
        'dsads': {
            'num_subjects':  8,
            'sampling_rate': 25,
            'file_extension': 'txt',
            'dataset_path':  '~/dataset/daily/daily/data',
            'imu_positions': ['chest', 'right_wrist', 'right_leg'],
            'column_prefix': ['chest', 'right_wrist', 'left_wrist', 'right_leg', 'left_leg'],
            'column_extra':  [],
            'column_drop':   None,
            'sensor_types':  ['accel', 'gyro', 'mag'],
            'n_samples':     60,
            'activities': {
                'sitting': 1, 'standing': 2, 'lying': 3, 'lying_right': 4,
                'AcdS': 5, 'DcdS': 6, 'standing_in_elevator_still': 7,
                'moving_around_elevator': 8, 'walking_park': 9, 'walking': 10,
                'nordic_walking': 11, 'running': 12, 'exercising_on_stepper': 13,
                'exercising_on_cross_trainer': 14, 'cycling_horizontal': 15,
                'cycling': 16, 'rowing': 17, 'jumping': 18, 'playing_basketball': 19,
            },
        },
    }

    def __init__(self):
        self.sensor_types       = ['accel', 'gyro', 'mag']
        self.compared_signal    = defaultdict(lambda: defaultdict(list))
        
    def _get_columns_lists(self, columns:list, positions:list, sensor_types:list) -> list:
        """ 
        Get the list of columns based on the positions and sensor types. 
        Args:
            columns (list): A list of column names in the dataframe.
            positions (list): A list of sensor positions to be included in the output tensor.
            sensor_types (list): A list of sensor types to be included in the output tensor.
        Returns:
            A list of column names that match the specified positions and sensor types.
        """
        if columns is None: columns = []
        
        for pos, data_type in product(positions, sensor_types):
            name = f'{pos}_{data_type}'
            if data_type == 'temp': columns.append(name)
            else:
                if data_type == 'orient':
                    columns.extend([f'{name}_{axis}' for axis in ['x', 'y', 'z', 'w']])
                else:
                    columns.extend([f'{name}_{axis}' for axis in ['x', 'y', 'z']])
        return columns
    
    def _select_class(self, activities_list:list) -> list:
        """ Select the classes based on the specified category. 
        Args:
            activities_list (list): A list of activity names in the dataset.
        Returns:
            A list of activity names that match the specified category.
        """
        if self.category == 'all':
            self.classes = list(activities_list.keys())
            
        elif self.category == 'custom':
            self.classes = [
                'sitting', 
                'standing', 
                'AcdS', 
                'walking',
                'lying',
                'running',
                'cycling'
            ]

    def load(self, dataset_name, category='all', positions=None, *args, **kwargs):
        if dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(f"Dataset '{dataset_name}' not supported. Choose from {list(self.DATASET_CONFIGS.keys())}")
        
        self.dataset_name    = dataset_name
        self.category        = category
        self.wanted_positions = positions
        self.kwargs          = kwargs
        
        return self._load_dataset(self.DATASET_CONFIGS[dataset_name])

    def _load_dataset(self, config):

        # Build column list
        columns = list(config['column_extra'])
        columns = self._get_columns_lists(columns, config['column_prefix'], config['sensor_types'])
        if self.dataset_name == 'mhealth':
            columns.append('label')

        # channels = config['sensor_types'] or self.data_types
        if 'sensor_types' in self.kwargs:
            
            if self.dataset_name == 'pamap2':
                data_types_arr = np.array(self.kwargs['sensor_types'])
                self.sensor_types = np.where(data_types_arr == 'accel', 'accel16', data_types_arr).tolist()
            else:
                self.sensor_types = self.kwargs['sensor_types']
        else:
            self.sensor_types = config['sensor_types']

        # Select classes
        self.activityIDdict = config['activities']
        self._select_class(self.activityIDdict)
        
        # Resolve imu positions
        self.imu_positions = config['imu_positions']
        if self.wanted_positions:
            # convert the wanted positions to match the dataset's column naming convention
            mapping = {
                'chest': ['chest'], 
                'hand': ['wrist', 'hand'],
                'ankle': ['leg', 'ankle'],
            }
            new_positions = []

            for wanted_pos in self.wanted_positions:
                for dataset_pos in self.imu_positions:
                    # Mapping the wanted position to the dataset position based on 
                    # the mapping dictionary and the column names in the dataset
                    if any(value in dataset_pos for value in mapping.get(wanted_pos, [])):
                        print(f'Mapping {wanted_pos} to {dataset_pos}')
                        new_positions.append(dataset_pos)
                        
            self.imu_positions = new_positions
            
        print(f'Activities:     {self.classes}')
        print(f'IMU positions:  {self.imu_positions}')
        print(f'Sensor types:   {self.sensor_types}')
        # remap the name of sensor

        raw_data = self._load(
            n_subjects    = config['num_subjects'],
            sampling      = config['sampling_rate'],
            loading_path  = config['dataset_path'],
            file_extension= config['file_extension'],
            columns       = columns,
            column_drop   = config.get('column_drop'),
            sampled       = config.get('n_samples'),
        )

        return {
            'dataset_name':  self.dataset_name,
            'data':          raw_data,
            'classes':       self.classes,
            'fs':            config['sampling_rate'],
            'imu_positions': self.imu_positions,
            'channels':      self.sensor_types,
        }
    
    def _load(self, 
              n_subjects:int, 
              sampling:int, 
              loading_path:str, 
              file_extension:str, 
              columns:list,
              column_drop:list=None,
              sampled:int=None,
              ):
        
        # Create the data collection structure
        raw_data = defaultdict(lambda: defaultdict(lambda: None))
        
        sep = {'log': '\s+',
                'txt': ',',
        }
        if sampled:
            # DSADS only for now
            for s in tqdm(range(n_subjects), desc=f'Loading {self.dataset_name} ', unit='subject'):
                for activity in tqdm(self.classes, desc=f"    └── Subject {s}", leave=False):
                    
                    activity_id = self.activityIDdict[activity]                 # Get the activity ID
                    if activity_id < 10: 
                        activity_id = f'0{activity_id}'        # Rename the file name
                    
                    # Collect the data of each sample
                    sampled_data = []
                    for n_sample in range(1, sampled+1):
                        if n_sample < 10:
                            n_sample = f'0{n_sample}'        # Rename the file name
                        
                        filename = f'{loading_path}/a{activity_id}/p{s+1}/s{n_sample}.{file_extension}'

                        dataframe = pd.read_csv(filename, sep=sep[file_extension], header=None)
                        dataframe.columns = columns
                        
                        data = self._rearange(dataframe, self.imu_positions, self.sensor_types)
                        sampled_data.append(data)
                        
                    raw_data[s][activity] = torch.stack(sampled_data, dim=0) # Shape: (n_samples, n_sensors, n_channels, times)
        else:
        
            for s in tqdm(range(n_subjects), desc=f'Loading {self.dataset_name} ', unit='subject'):
                # Get the filename
                filename    = f'{loading_path}{s+1}.{file_extension}'
                if file_extension == 'dat':
                    dataframe = pd.read_table(filename, sep=r'\s+', header=None)
                else:
                    dataframe = pd.read_csv(filename, sep=sep[file_extension], header=None)
                dataframe.columns = columns
                
                for activity in tqdm(self.classes, desc=f"    └── Subject {s}", leave=False):
                    activity_id = self.activityIDdict[activity]
                    df = dataframe[dataframe['label'] == activity_id]   # get wanted activity
                    if column_drop is not None:
                        df = df.drop(column_drop, axis='columns')
                    
                    # Remove the first and last 5 seconds of the activity to avoid transition states
                    df = df.iloc[5 * sampling: -5*sampling].reset_index(drop=True)
                    
                    if self.dataset_name == 'pamap2':
                        if df.empty or len(df) < 12 * sampling:
                            continue
                    
                    # Rearange the dataframe to a tensor of shape (num_sensors, num_channels, times)
                    raw_data[s][activity] = self._rearange(df, self.imu_positions, self.sensor_types)
                    
        return raw_data
                
    def _rearange(self, dataframe, positions, channels) -> torch.Tensor:
        """ Rearrange the dataframe to a tensor of shape 
        Args:
            dataframe (pd.DataFrame): The input dataframe containing the sensor data.
            positions (list): A list of sensor positions to be included in the output tensor.
            channels (list): A list of sensor channels to be included in the output tensor.
        Returns:
            (num_sensors, num_channels, times) 
        """
        data = []
        
        for pos_idx, pos_name in enumerate(positions):
            # Select the conlumns based on the position and channel names
            selected_column = [
                col for col in dataframe.columns 
                if any(channel in col for channel in channels) and pos_name in col
            ]
            # Exclude the magnetometer data
            selected_column = [col for col in selected_column if not 'mag' in col]

            # Get the data following the selected columns
            df = dataframe[selected_column]
            # Interpolate the missing values
            df = df.interpolate(method='linear', limit_direction='both')
            
            # Convert to tensor and rearrange the dimensions
            tensor = torch.from_numpy(df.to_numpy())
            tensor = tensor.transpose(0, 1) # Shape: (num_channels, times)
            
            # Append the tensor to the list
            data.append(tensor.unsqueeze(0)) # Shape: (1, num_channels, times)

        data = torch.cat(data, dim=0) # Shape: (num_sensors, num_channels, times)
        return data