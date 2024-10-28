import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from omegaconf import DictConfig

MODALITY_MAPPING = {    
    'local_representation': 'mano',
    'faive_angles': 'faive',
    'simple_gripper': 'simple_gripper'
}

def adapt_data_keys(data, modality_mapping=MODALITY_MAPPING):
    return {modality_mapping[key]: data[key] for key in data.keys()}

def normalize_data(data, normalization_params):
    normalized_data = {}
    for key, tensor in data.items():
        mean = normalization_params[key]['mean']
        std = normalization_params[key]['std']
        normalized_data[key] = (tensor - mean) / (std + 1e-8)
    return normalized_data

def denormalize_data(data, normalization_params):
    denormalized_data = {}
    for key, tensor in data.items():
        mean = normalization_params[key]['mean']
        std = normalization_params[key]['std']
        denormalized_data[key] = tensor * std + mean
    return denormalized_data

class RobotClipDataset(Dataset):
    def __init__(self, config: DictConfig):
        self.data = np.load(config.data.source_file, allow_pickle=True).item()
        self.modality_mapping = MODALITY_MAPPING
        self.data = adapt_data_keys(self.data, self.modality_mapping)
        self.normalization_params = self.compute_normalization_params(self.data)
        self.normalized_data = normalize_data(self.data, self.normalization_params)
    
    def compute_normalization_params(self, data):
        normalization_params = {}
        for key, tensor in data.items():
            mean = np.mean(tensor, axis=0)
            std = np.std(tensor, axis=0)
            normalization_params[key] = {'mean': mean, 'std': std}

        return normalization_params

    def __len__(self):
        return len(next(iter(self.normalized_data.values())))

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.normalized_data.items()}

def load_data(config: DictConfig):
    dataset = RobotClipDataset(config)
    return dataset.normalized_data, dataset.normalization_params

def get_dataloaders(config: DictConfig):
    dataset = RobotClipDataset(config)

    # Split the dataset
    train_size = int(config.data.train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size)

    return train_loader, test_loader, dataset.normalization_params
