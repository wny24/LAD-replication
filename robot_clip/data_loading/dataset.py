import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from omegaconf import DictConfig

MODALITY_MAPPING = {
    "local_representation": "mano",
    "faive_angles": "faive",
    "simple_gripper": "simple_gripper",
    "xhand_angles": "xhand",
    "g2_width": "g2",
}

def adapt_data_keys(data, modality_mapping=MODALITY_MAPPING):
    return {modality_mapping[key]: data[key] for key in data.keys()}

def _match_stats(stat, tensor):
    """Cast mean/std to the same type and device as the data array."""
    if isinstance(tensor, torch.Tensor):
        return torch.as_tensor(stat, dtype=tensor.dtype, device=tensor.device)
    return np.asarray(stat, dtype=np.asarray(tensor).dtype)

def normalize_data(data, normalization_params):
    normalized_data = {}
    for key, tensor in data.items():
        mean = _match_stats(normalization_params[key]['mean'], tensor)
        std = _match_stats(normalization_params[key]['std'], tensor)
        normalized_data[key] = (tensor - mean) / (std + 1e-8)
    return normalized_data

def denormalize_data(data, normalization_params):
    denormalized_data = {}
    for key, tensor in data.items():
        mean = _match_stats(normalization_params[key]['mean'], tensor)
        std = _match_stats(normalization_params[key]['std'], tensor)
        denormalized_data[key] = tensor * std + mean
    return denormalized_data

class RobotClipDataset(Dataset):
    def __init__(self, config: DictConfig):
        self.data = np.load(config.data.source_file, allow_pickle=True).item()
        self.modality_mapping = MODALITY_MAPPING
        self.data = adapt_data_keys(self.data, self.modality_mapping)
        self.data = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in self.data.items()}
        self.normalization_params = self.compute_normalization_params(self.data)
        self.normalized_data = normalize_data(self.data, self.normalization_params)
    
    def compute_normalization_params(self, data):
        normalization_params = {}
        for key, tensor in data.items():
            arr = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
            mean = np.mean(arr, axis=0)
            std = np.std(arr, axis=0)
            mins = np.min(arr, axis=0)
            maxs = np.max(arr, axis=0)
            normalization_params[key] = {'mean': mean, 'std': std, 'mins': mins, 'maxs': maxs}

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
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, pin_memory=True, num_workers=4)

    return train_loader, test_loader, dataset.normalization_params
