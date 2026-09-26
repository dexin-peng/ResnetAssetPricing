import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from functools import partial


class FinancialDataset(Dataset):
    def __init__(self, n):
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        return idx

def index_collate(batch_idx, X_tensor, y_tensor):
    idx = torch.as_tensor(batch_idx, device=X_tensor.device, dtype=torch.long)
    return (
        X_tensor.index_select(0, idx),
        y_tensor.index_select(0, idx)
    )

class FinancialDataloader():
    def __init__(self, config, now, X=None, y=None):
        self.config = config
        self.now = now
        self.X = X
        self.y = y
        self.config.data.split_time_window(self.now, self.config.data.time_loop_freq)
        if self.config.data.use_pre_split_dataset: self._load_pre_split_dataset()
        if not self.config.data.use_pre_split_dataset and not self.config.data.init_pre_split_dataset: self._ignore_pre_split_dataset_loading()
        if self.config.data.init_pre_split_dataset: self._init_pre_split_dataset()
        if not self.config.data.init_pre_split_dataset: self._load_dataloader()

    def _load_dataloader(self):
        if self.config.data.train:
            self.train_loader = DataLoader(FinancialDataset(len(self.X_train_tensor)), batch_size=self.config.training.batch_size, shuffle=False, collate_fn=partial(index_collate, X_tensor=self.X_train_tensor, y_tensor=self.y_train_tensor), num_workers=0)
            self.early_stopping_loader = DataLoader(FinancialDataset(len(self.X_early_stopping_tensor)), batch_size=self.config.training.batch_size, shuffle=False, collate_fn=partial(index_collate, X_tensor=self.X_early_stopping_tensor, y_tensor=self.y_early_stopping_tensor), num_workers=0)
        self.test_loader = DataLoader(FinancialDataset(len(self.X_test_tensor)), batch_size=self.config.training.batch_size, shuffle=False, collate_fn=partial(index_collate, X_tensor=self.X_test_tensor, y_tensor=self.y_test_tensor), num_workers=0)

    def _init_pre_split_dataset(self):
        self.X_train, self.X_test, \
        self.y_train, self.y_test \
            = self.config.data.split_data(self.X, self.y, now=self.now, freq=self.config.data.time_loop_freq)
        self.X_early_stopping, self.y_early_stopping = self.config.data.split_early_stopping_data(self.X_train, self.y_train)
        self.X_train_tensor = torch.from_numpy(self.X_train.values).float()
        self.y_train_tensor = torch.from_numpy(self.y_train.values).float().unsqueeze(1)
        self.X_early_stopping_tensor = torch.from_numpy(self.X_early_stopping.values).float()
        self.y_early_stopping_tensor = torch.from_numpy(self.y_early_stopping.values).float().unsqueeze(1)
        self.X_test_tensor = torch.from_numpy(self.X_test.values).float()
        self.y_test_tensor = torch.from_numpy(self.y_test.values).float().unsqueeze(1)

        torch.save(self.X_train_tensor, self.config.io.X_train_tensor_path(self.now))
        pd.to_pickle(self.X_train.index, self.config.io.X_train_index_path(self.now))
        torch.save(self.y_train_tensor, self.config.io.y_train_tensor_path(self.now))
        pd.to_pickle(self.y_train.index, self.config.io.y_train_index_path(self.now))
        torch.save(self.X_test_tensor, self.config.io.X_test_tensor_path(self.now))
        pd.to_pickle(self.X_test.index, self.config.io.X_test_index_path(self.now))
        torch.save(self.y_test_tensor, self.config.io.y_test_tensor_path(self.now))
        pd.to_pickle(self.y_test.index, self.config.io.y_test_index_path(self.now))

        pd.to_pickle(self.X_test.columns, self.config.io.X_columns_path(self.now))

        self.config.logger.info(f"Pre split dataset for now={self.now.strftime('%Y-%m-%d')} saved to {self.config.io.temp_dir}")
        del self.X, self.y
        del self.X_train, self.X_test, self.y_train, self.y_test
        del self.X_train_tensor, self.X_test_tensor, self.y_train_tensor, self.y_test_tensor
    
    def _ignore_pre_split_dataset_loading(self):
        self.X_train, self.X_test, \
        self.y_train, self.y_test \
            = self.config.data.split_data(self.X, self.y, now=self.now, freq=self.config.data.time_loop_freq)
        self.X_early_stopping, self.y_early_stopping = self.config.data.split_early_stopping_data(self.X_train, self.y_train)
        self.X_train_index = self.X_train.index
        self.y_train_index = self.y_train.index
        self.X_test_index = self.X_test.index
        self.y_test_index = self.y_test.index
        self.X_train_tensor = torch.from_numpy(self.X_train.values).float().to(self.config.data.device)
        self.y_train_tensor = torch.from_numpy(self.y_train.values).float().unsqueeze(1).to(self.config.data.device)
        self.X_early_stopping_tensor = torch.from_numpy(self.X_early_stopping.values).float().to(self.config.data.device)
        self.y_early_stopping_tensor = torch.from_numpy(self.y_early_stopping.values).float().unsqueeze(1).to(self.config.data.device)
        self.X_test_tensor = torch.from_numpy(self.X_test.values).float().to(self.config.data.device)
        self.y_test_tensor = torch.from_numpy(self.y_test.values).float().unsqueeze(1).to(self.config.data.device)

    def _load_pre_split_dataset(self):
        if self.config.data.train:
            self.X_train_tensor = torch.load(self.config.io.X_train_tensor_path(self.now), map_location=self.config.data.device)
            self.X_train_index = pd.read_pickle(self.config.io.X_train_index_path(self.now))
            self.y_train_tensor = torch.load(self.config.io.y_train_tensor_path(self.now), map_location=self.config.data.device)
            self.y_train_index = pd.read_pickle(self.config.io.y_train_index_path(self.now))
            self._build_early_stopping_tensors_from_pre_split_train()

        self.X_test_tensor = torch.load(self.config.io.X_test_tensor_path(self.now), map_location=self.config.data.device)
        self.X_test_index = pd.read_pickle(self.config.io.X_test_index_path(self.now))
        self.y_test_tensor = torch.load(self.config.io.y_test_tensor_path(self.now), map_location=self.config.data.device)
        self.y_test_index = pd.read_pickle(self.config.io.y_test_index_path(self.now))

    def _build_early_stopping_tensors_from_pre_split_train(self):
        if 'DATE' in self.X_train_index.names:
            train_dates = pd.to_datetime(self.X_train_index.get_level_values('DATE'))
        else:
            train_dates = pd.to_datetime(self.X_train_index.get_level_values(-1))
        mask = (train_dates >= self.config.data.early_stopping_start) & (train_dates <= self.config.data.early_stopping_end)
        early_stopping_idx = torch.as_tensor(np.flatnonzero(np.asarray(mask)), device=self.X_train_tensor.device, dtype=torch.long)
        self.X_early_stopping_tensor = self.X_train_tensor.index_select(0, early_stopping_idx)
        self.y_early_stopping_tensor = self.y_train_tensor.index_select(0, early_stopping_idx)
