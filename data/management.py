# data folder: management.py
from torch.utils.data import Dataset, Subset, DataLoader
import torchaudio.transforms as T

from torch.fft import *
import random
from sklearn.manifold import TSNE
import torch
import os

import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# Function to add Gaussian noise
class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0.0, std=0.01):
        super(AddGaussianNoise, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise


class CustomDataset(Dataset):
    def __init__(self, X_train, y_train, use_transform=False, target_transform=None, num_masks=2):
        self.X_train = X_train
        self.y_train = y_train
        self.use_transform = use_transform

        self.target_transform = target_transform
        self.transform_fn = torch.nn.Sequential(
            T.TimeMasking(time_mask_param=10, p=1.0),
        )
        self.min_mask   = 5
        self.max_mask   = 10
        self.num_masks  = num_masks
        self.__print = False
        
    def random_time_mask(self, x):
        """Apply multiple random time masks to a spectrogram."""
        
        for _ in range(self.num_masks):
            param = random.randint(self.min_mask, self.max_mask)
            time_masking = T.TimeMasking(time_mask_param=param, p=1.0)
            x = time_masking(x)
        return x

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        
        X       = self.X_train[idx]
        # X = self.transform_fn(X) 
        label   = self.y_train[idx]

        if self.use_transform:

            # Make sure X is a torch.Tensor
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)

            # Apply time masking (must be float & spectrogram-like input)
            X = self.random_time_mask(X)

        if self.target_transform:   
            label = self.target_transform(label)

        return X, label
    
    def get_sample_shape(self):
        """
        Return the shape of the first sample in the dataset.
        """
        sample = self.X_train[0]  # Get the first sample
        return sample.shape if isinstance(sample, torch.Tensor) else len(sample)
    
    def get_unique_labels(self):
        """
        Returns unique labels in the dataset.
        """
        if isinstance(self.y_train, torch.Tensor):
            return torch.unique(self.y_train).tolist()
        else:  # If y_train is a list or NumPy array
            return list(set(self.y_train))
    
    def get_data_shape(self):
        return self.X_train[0].shape
            

def split_dataset(dataset, train_test_split=0.8):
    dataset_length  = len(dataset)
    train_length    = int(dataset_length * train_test_split)
    test_length     = dataset_length - train_length

    test_indices   = random.sample(range(dataset_length), test_length)
    train_indices  = [i for i in range(dataset_length) if i not in test_indices]
    
    train_dataset   = Subset(dataset, train_indices)
    test_dataset    = Subset(dataset, test_indices)
    

    return train_dataset, test_dataset


def create_real_test_dataset(raw, labels:list, seed:int=32):
    training_tasks = list(raw.keys())  
    dataset = {}
    activities = labels
    n_samples = 0
    for s in training_tasks:
        x = []
        y = []
        for id, activity in enumerate(activities):
            if raw[s][activity] is None:
                continue
            data = raw[s][activity]
            x.append(data)
            y.append(torch.zeros(data.size(0), dtype=torch.long) + id)
        x = torch.cat(x, dim=0).to(torch.float32)
        y = torch.cat(y, dim=0)
        n_samples += int(x.size(0))
        dataset[s] = CustomDataset(x, y, use_transform=False)
    print(f'Total test samples: {n_samples}')
    return dataset

                
def build_dataset(dataset, seed=0, stage='train', **kwargs) -> tuple:
    """
    Build the training dataset by concatenating data from all subjects and activities.
    Args:
        dataset (dict):     The input dataset containing data for each subject and activity.
        is_shuffle (bool):  Whether to shuffle the dataset after concatenation.
        seed (int):         Random seed for shuffling the dataset.
    """
    
    X = []
    Y = []
    subject_count = {}
    activity_to_id = {act: i for i, act in enumerate(dataset['classes'])}
    
    for subject, activities in dataset['data'].items():
        x = []
        y = []
        collected_activities = []
        act_ID = 0
        count_samples = 0
        
        for activity, data in activities.items():

            if data is None: 
                continue
            if activity not in collected_activities:
                if len(collected_activities) > 0:
                    act_ID += 1
                collected_activities.append(activity)
                
            x.append(data)
            y.extend(torch.full((data.size(0),), activity_to_id[activity], dtype=torch.long))
            count_samples += data.size(0)
            
        X.append(torch.cat(x, dim=0).to(torch.float32))
        Y.append(torch.tensor(y))
        subject_count[subject] = count_samples
    
    X = torch.cat(X, dim=0)
    Y = torch.cat(Y, dim=0)
    
    
    if stage == 'test':
        return None, (X, Y), subject_count
    else:
        x_train, y_train = shuffle(X, Y, random_state=seed)
        if 'val_split' in kwargs:
            x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=kwargs['val_split'], random_state=seed)
            print(f'Training samples: {len(x_tr)}, Validation samples: {len(x_val)}')
            return (x_tr, y_tr), (x_val, y_val), None
        
        return (x_train, y_train), None, None


if __name__ == '__main__':
    path = os.getcwd()

