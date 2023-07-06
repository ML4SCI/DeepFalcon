import torch
from torch.utils.data import *

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

    def analyze_data(self):
        num_samples = len(self)
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        track_means = []
        ecal_means = []
        hcal_means = []
        track_mins = []
        ecal_mins = []
        hcal_mins = []
        track_maxs = []
        ecal_maxs = []
        hcal_maxs = []

        for batch_idx in range(num_batches):
            start_index = batch_idx * self.batch_size
            end_index = min(start_index + self.batch_size, num_samples)
            batch_data = self.get_batch(start_index, end_index)
            track_means.append(np.mean(batch_data[:, :, 0]))
            ecal_means.append(np.mean(batch_data[:, :, 1]))
            hcal_means.append(np.mean(batch_data[:, :, 2]))
            track_mins.append(np.min(batch_data[:, :, 0]))
            ecal_mins.append(np.min(batch_data[:, :, 1]))
            hcal_mins.append(np.min(batch_data[:, :, 2]))
            track_maxs.append(np.max(batch_data[:, :, 0]))
            ecal_maxs.append(np.max(batch_data[:, :, 1]))
            hcal_maxs.append(np.max(batch_data[:, :, 2]))

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