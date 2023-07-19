import torch
from torch.utils.data import *
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ParquetDataset(Dataset):
    def __init__(self, filename, batch_size=256):
        self.parquet = pq.ParquetFile(filename)
        self.cols = None
        self.verbose = False
        self.batch_size = batch_size

    def __getitem__(self, index):
        data = self.parquet.read_row_group(index, columns=self.cols).to_pydict()
        data['X_jets'] = np.float32(data['X_jets'][0])
        data['X_jets'] = data['X_jets'][0:]

        reshaped_data = np.transpose(data['X_jets'], (1, 2, 0))

        return reshaped_data

    def __len__(self):
        return self.parquet.num_row_groups

    def get_batch(self, start_index, end_index):
        data = self.parquet.read_row_group(start_index, columns=self.cols).to_pydict()
        data['X_jets'] = np.float32(data['X_jets'][0])
        data['X_jets'] = data['X_jets'][0:]

        reshaped_data = np.transpose(data['X_jets'], (1, 2, 0))

        return reshaped_data


class PointParquetDataset(Dataset):
    def __init__(self, filename, batch_size=256, transform=None):
        self.parquet = pq.ParquetFile(filename)
        self.cols = None
        self.verbose = False
        self.batch_size = batch_size
        self.transform = transform

    def __getitem__(self, index):
        data = self.parquet.read_row_group(index, columns=self.cols).to_pydict()
        data['X_jets'] = np.float32(data['X_jets'][0])
        # print(data['X_jets'].shape)
        # print(data['X_jets'][0:].shape)
        image = data['X_jets'][0:]
        image = np.transpose(image, (1, 2, 0))

        non_zero_Tracks = np.nonzero(image[:, :, 0])
        non_zero_ECAL = np.nonzero(image[:, :, 1])
        non_zero_HCAL = np.nonzero(image[:, :, 2])
        coords_Tracks = np.column_stack(non_zero_Tracks)
        coords_ECAL = np.column_stack(non_zero_ECAL)
        coords_HCAL = np.column_stack(non_zero_HCAL)

        #For visualization placing Tracks, ECAL, HCAL on z = 0,1,2 respectively. However when training 2D surface would be used i.e z=0 for all channels
        values_Tracks = image[non_zero_Tracks[0], non_zero_Tracks[1], 0]     
        values_ECAL = image[non_zero_ECAL[0], non_zero_ECAL[1], 1] 
        values_HCAL = image[non_zero_HCAL[0], non_zero_HCAL[1], 2]

        coords_Tracks = np.hstack((coords_Tracks, np.zeros((coords_Tracks.shape[0], 1))))
        coords_ECAL = np.hstack((coords_ECAL, np.zeros((coords_ECAL.shape[0], 1))))
        coords_HCAL = np.hstack((coords_HCAL, np.zeros((coords_HCAL.shape[0], 1))))
        
        # Store the point cloud for this image in the list
        point_cloud = {'tracks': (coords_Tracks, values_Tracks), 'ECAL': (coords_ECAL, values_ECAL), 'HCAL': (coords_HCAL, values_HCAL)}

        return point_cloud

    def __len__(self):
        return self.parquet.num_row_groups


def compute_mean_max_min(dataset):
    num_samples = len(dataset)
    track_max_list = []
    ecal_max_list = []
    hcal_max_list = []

    track_min_list = []
    ecal_min_list = []
    hcal_min_list = []

    for index in range(num_samples):
        data = dataset[index]
        # Track channel
        track_max = np.max(data[:, :, 0])
        track_min = np.min(data[:, :, 0])
        track_max_list.append(track_max)
        track_min_list.append(track_min)

        # ECAL channel
        ecal_max = np.max(data[:, :, 1])
        ecal_min = np.min(data[:, :, 1])
        ecal_max_list.append(ecal_max)
        ecal_min_list.append(ecal_min)

        # HCAL channel
        hcal_max = np.max(data[:, :, 2])
        hcal_min = np.min(data[:, :, 2])
        hcal_max_list.append(hcal_max)
        hcal_min_list.append(hcal_min)

    # Compute the mean values
    track_mean_max = np.mean(track_max_list)
    ecal_mean_max = np.mean(ecal_max_list)
    hcal_mean_max = np.mean(hcal_max_list)

    track_mean_min = np.mean(track_min_list)
    ecal_mean_min = np.mean(ecal_min_list)
    hcal_mean_min = np.mean(hcal_min_list)

    return {
        'track_mean_max': track_mean_max,
        'ecal_mean_max': ecal_mean_max,
        'hcal_mean_max': hcal_mean_max,
        'track_mean_min': track_mean_min,
        'ecal_mean_min': ecal_mean_min,
        'hcal_mean_min': hcal_mean_min
    }

def analyze_data(dataset):
    num_samples = len(dataset)
    track_means = []
    ecal_means = []
    hcal_means = []
    track_mins = []
    ecal_mins = []
    hcal_mins = []
    track_maxs = []
    ecal_maxs = []
    hcal_maxs = []

    for index in range(num_samples):
        data = dataset[index]
        track_means.append(np.mean(data[:, :, 0]))
        ecal_means.append(np.mean(data[:, :, 1]))
        hcal_means.append(np.mean(data[:, :, 2]))
        track_mins.append(np.min(data[:, :, 0]))
        ecal_mins.append(np.min(data[:, :, 1]))
        hcal_mins.append(np.min(data[:, :, 2]))
        track_maxs.append(np.max(data[:, :, 0]))
        ecal_maxs.append(np.max(data[:, :, 1]))
        hcal_maxs.append(np.max(data[:, :, 2]))

    track_mean = np.mean(track_means)
    ecal_mean = np.mean(ecal_means)
    hcal_mean = np.mean(hcal_means)
    track_min = np.min(track_mins)
    ecal_min = np.min(ecal_mins)
    hcal_min = np.min(hcal_mins)
    track_max = np.max(track_maxs)
    ecal_max = np.max(ecal_maxs)
    hcal_max = np.max(hcal_maxs)

    return {
        'track_mean': track_mean,
        'ecal_mean': ecal_mean,
        'hcal_mean': hcal_mean,
        'track_min': track_min,
        'ecal_min': ecal_min,
        'hcal_min': hcal_min,
        'track_max': track_max,
        'ecal_max': ecal_max,
        'hcal_max': hcal_max
    }

