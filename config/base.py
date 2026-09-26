import os
from pathlib import Path
import ml_collections
import pandas as pd
import torch
from pandas import IndexSlice as idx
from pandas.tseries.offsets import MonthEnd, YearEnd, YearBegin
from datetime import datetime
import numpy as np


class Config(ml_collections.ConfigDict):
    def __init__(self, args):
        super().__init__()
        self.training = ml_collections.ConfigDict()
        self.training.continue_training = True
        self.training.batch_size = args.batch_size
        self.training.epochs = args.epochs
        self.training.early_stopping_patience = args.early_stopping_patience
        self.training.l1 = args.l1
        self.training.l1_lambda = args.l1_lambda
        self.training.l2 = args.l2
        self.training.optimizer = args.optimizer
        self.training.weight_decay = args.weight_decay
        self.training.lr = args.initial_lr
        self.training.record_grad = args.record_grad
        self.training.snapshot_freq = 50
        self.training.device = args.device
        self.econ = ml_collections.ConfigDict()
        self.econ.econ_mkt_cap = args.econ_mkt_cap
        self.econ.econ_weight_type = args.econ_weight_type
        self.data = DataConfig(args)
        self.data.init_pre_split_dataset = args.init_pre_split_dataset
        self.data.use_pre_split_dataset = args.use_pre_split_dataset
        self.data.train = args.train
        self.data.predict = args.predict
        self.model = ml_collections.ConfigDict()
        self.model.start_ensemble = args.start_ensemble
        self.model.max_ensemble = args.max_ensemble

        # input and output specification will be set in each model config
        self.io = IOConfig(model_name=None, asset_prefix=None)

class IOConfig(ml_collections.ConfigDict):
    def __init__(self, model_name, asset_prefix):
        super().__init__()
        '''
        class properties:
        
        - model_name: the name of the model, set in each model config

        - data_dir: the directory of the data path
        - price_characteristics: the name of the price characteristics data
        - macro_predictors: the name of the macro predictors data file
        - data_description: the name of the data description file
        - clean_X_path: the path of the cleaned X data
        - clean_y_path: the path of the cleaned y data
        - tensorboard_dir: the directory of the tensorboard
        - temp_dir: the directory of the temporary files

        F is for Function
        - checkpoint_dir: the directory of the checkpoints
        - F: checkpoint_dir_NE: the directory of the checkpoint for the specific Now and Ensemble
        - F: checkpoint_NE: the file path of the checkpoint for the specific Now and Ensemble and epoch
        - F: latest_epoch_NE: the latest epoch of the checkpoint for the specific Now and Ensemble
        - F: latest_checkpoint_NE: the file path of the latest checkpoint of the checkpoint for the specific Now and Ensemble

        - predicts_dir: the directory of the predicts files
        - F: predicts_NE: the file path of the predicts for the specific Now and Ensemble

        - log_dir: the directory of the log files

        '''
        self.model_name = model_name
        self.asset_prefix = asset_prefix
        self.data_dir = 'source_data'
        self.price_characteristics = os.path.join(self.data_dir, 'datashare_with_return.pkl')
        self.macro_predictors = os.path.join(self.data_dir, 'PredictorData2024.xlsx')
        self.data_description = os.path.join(self.data_dir, 'Company94Characteristics.csv')
        self.clean_X_path = os.path.join(self.temp_dir, 'clean_X.pkl')
        self.clean_y_path = os.path.join(self.temp_dir, 'clean_y.pkl')
        self.clean_mvel1_path = os.path.join(self.temp_dir, 'clean_mvel1.pkl')
        self.clean_ff5_path = os.path.join(self.temp_dir, 'clean_ff5.pkl')


    @property
    def base_dir(self):
        path = str(Path(__file__).resolve().parents[1])
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def asset_dir(self):
        path = os.path.join(self.base_dir, 'asset', self.asset_prefix)
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def tensorboard_dir(self):
        path = os.path.join(self.asset_dir, 'tensorboard')
        os.makedirs(path, exist_ok=True)
        return path
    
    @property
    def temp_dir(self):
        path = os.path.join(self.base_dir, 'tmp')
        os.makedirs(path, exist_ok=True)
        return path

    def X_train_tensor_path(self, now):
        path = os.path.join(self.temp_dir, now.strftime('%Y%m%d'), 'X_train_tensor.pt')
        os.makedirs(os.path.join(self.temp_dir, now.strftime('%Y%m%d')), exist_ok=True)
        return path

    def X_columns_path(self, now):
        path = os.path.join(self.temp_dir, now.strftime('%Y%m%d'), 'X_columns.pkl')
        os.makedirs(os.path.join(self.temp_dir, now.strftime('%Y%m%d')), exist_ok=True)
        return path

    def X_train_index_path(self, now):
        path = os.path.join(self.temp_dir, now.strftime('%Y%m%d'), 'X_train_index.pkl')
        os.makedirs(os.path.join(self.temp_dir, now.strftime('%Y%m%d')), exist_ok=True)
        return path

    def y_train_tensor_path(self, now):
        path = os.path.join(self.temp_dir, now.strftime('%Y%m%d'), 'y_train_tensor.pt')
        os.makedirs(os.path.join(self.temp_dir, now.strftime('%Y%m%d')), exist_ok=True)
        return path

    def y_train_index_path(self, now):
        path = os.path.join(self.temp_dir, now.strftime('%Y%m%d'), 'y_train_index.pkl')
        os.makedirs(os.path.join(self.temp_dir, now.strftime('%Y%m%d')), exist_ok=True)
        return path

    def X_test_tensor_path(self, now):
        path = os.path.join(self.temp_dir, now.strftime('%Y%m%d'), 'X_test_tensor.pt')
        os.makedirs(os.path.join(self.temp_dir, now.strftime('%Y%m%d')), exist_ok=True)
        return path
    
    def X_test_index_path(self, now):
        path = os.path.join(self.temp_dir, now.strftime('%Y%m%d'), 'X_test_index.pkl')
        os.makedirs(os.path.join(self.temp_dir, now.strftime('%Y%m%d')), exist_ok=True)
        return path
    
    def y_test_tensor_path(self, now):
        path = os.path.join(self.temp_dir, now.strftime('%Y%m%d'), 'y_test_tensor.pt')
        os.makedirs(os.path.join(self.temp_dir, now.strftime('%Y%m%d')), exist_ok=True)
        return path

    def y_test_index_path(self, now):
        path = os.path.join(self.temp_dir, now.strftime('%Y%m%d'), 'y_test_index.pkl')
        os.makedirs(os.path.join(self.temp_dir, now.strftime('%Y%m%d')), exist_ok=True)
        return path

    @property
    def econ_dir(self):
        path = os.path.join(self.asset_dir, 'econ')
        os.makedirs(path, exist_ok=True)
        return path
    
    def econ_brief_results_path(self, prefix):
        name = 'brief_results.csv'
        os.makedirs(os.path.join(self.econ_dir, prefix), exist_ok=True)
        return os.path.join(self.econ_dir, prefix, name)
    
    def econ_vw_decile_SR_table_path(self, prefix):
        name = 'econ_vw_decile_SR_table.csv'
        os.makedirs(os.path.join(self.econ_dir, prefix), exist_ok=True)
        return os.path.join(self.econ_dir, prefix, name)
    
    def econ_ew_decile_SR_table_path(self, prefix):
        name = 'econ_ew_decile_SR_table.csv'
        os.makedirs(os.path.join(self.econ_dir, prefix), exist_ok=True)
        return os.path.join(self.econ_dir, prefix, name)

    def econ_vw_decile_return_path(self, prefix):
        name = 'econ_vw_decile_return.csv'
        os.makedirs(os.path.join(self.econ_dir, prefix), exist_ok=True)
        return os.path.join(self.econ_dir, prefix, name)

    def econ_vw_turnover_path(self, prefix):
        name = 'econ_vw_turnover.csv'
        os.makedirs(os.path.join(self.econ_dir, prefix), exist_ok=True)
        return os.path.join(self.econ_dir, prefix, name)

    def econ_vw_decile_return_plot_path(self, prefix):
        name = 'econ_vw_decile_return.pdf'
        os.makedirs(os.path.join(self.econ_dir, prefix), exist_ok=True)
        return os.path.join(self.econ_dir, prefix, name)

    def econ_ew_decile_return_path(self, prefix):
        name = 'econ_ew_decile_return.csv'
        os.makedirs(os.path.join(self.econ_dir, prefix), exist_ok=True)
        return os.path.join(self.econ_dir, prefix, name)

    def econ_ew_turnover_path(self, prefix):
        name = 'econ_ew_turnover.csv'
        os.makedirs(os.path.join(self.econ_dir, prefix), exist_ok=True)
        return os.path.join(self.econ_dir, prefix, name)

    def econ_ew_decile_return_plot_path(self, prefix):
        name = 'econ_ew_decile_return.pdf'
        os.makedirs(os.path.join(self.econ_dir, prefix), exist_ok=True)
        return os.path.join(self.econ_dir, prefix, name)

    @property
    def checkpoint_dir(self):
        path = os.path.join(self.asset_dir, 'ckpt')
        os.makedirs(path, exist_ok=True)
        return path

    def checkpoint_dir_NE(self, now, ensemble):
        ckpt_path = os.path.join(self.checkpoint_dir, now.strftime('%Y%m%d'))
        path = os.path.join(ckpt_path, f'ensemble_{ensemble}')
        os.makedirs(path, exist_ok=True)
        return path
    
    def checkpoint_NE(self, now, epoch, ensemble):
        return os.path.join(self.checkpoint_dir_NE(now, ensemble), f'epoch_{epoch}.pth')
    
    def latest_epoch_NE(self, now, ensemble):
        checkpoint_dir = self.checkpoint_dir_NE(now, ensemble)
        if not os.path.exists(checkpoint_dir):
            return None
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('epoch_') and f.endswith('.pth')]
        if not checkpoints:
            return None
        latest_epoch = max([int(f.split('_')[1].split('.')[0]) for f in checkpoints])
        return latest_epoch

    def latest_checkpoint_NE(self, now, ensemble):
        if self.latest_epoch_NE(now, ensemble) is None:
            return None
        return os.path.join(self.checkpoint_dir_NE(now, ensemble), f'epoch_{self.latest_epoch_NE(now, ensemble)}.pth')

    @property
    def predicts_dir(self):
        path = os.path.join(self.asset_dir, 'predicts')
        os.makedirs(path, exist_ok=True)
        return path

    def predicts_NE(self, now, ensemble):
        os.makedirs(os.path.join(self.predicts_dir, now.strftime('%Y%m%d')), exist_ok=True)
        path = os.path.join(self.predicts_dir, now.strftime('%Y%m%d'), f'predicts_{ensemble}.pkl')
        return path

    @property
    def log_dir(self):
        path = os.path.join(self.asset_dir, 'log')
        os.makedirs(path, exist_ok=True)
        return path

class DataConfig(ml_collections.ConfigDict):
    def __init__(self, args):
        super().__init__()
        self.number_of_chars = 153
        self.number_of_macro = 8
        self.source_data_start = datetime(1963, 1, 31)
        self.source_data_end = datetime(2023, 12, 31)
        self.macro_M_sheet = 'Monthly'
        self.macro_Q_sheet = 'Quarterly'
        self.macro_A_sheet = 'Annual'
        self.selected_macro = ['d/p', 'e/p', 'b/m', 'ntis', 'tbl', 'tms', 'dfy', 'svar']
        self.device = args.device
        self.train_start = datetime(1963, 1, 31)
        self.time_loop_start = datetime(1987, 1, 31)
        self.time_loop_end  = self.source_data_end
        self.time_loop_freq = 'YE'
        self.time_loop_freq_dt = YearEnd(1)
        self.early_stopping_window_years = 5

    def split_time_window(self, now, freq=None):
        freq = self.time_loop_freq
        now = pd.to_datetime(now)

        if freq == 'YE':
            # Use the latest calendar year ending on or before now as the test window.
            if now.year == (now + YearEnd(1)).year:
                self.test_end = now - YearEnd(1)
            else:
                self.test_end = now

            self.train_end = self.test_end - YearEnd(1)
            self.test_start = self.test_end - YearBegin(1) + MonthEnd(1)
            self.early_stopping_end = self.train_end
            self.early_stopping_start = self.early_stopping_end - YearEnd(self.early_stopping_window_years) + MonthEnd(1)
            if self.early_stopping_start < self.train_start:
                self.early_stopping_start = self.train_start


    def split_data(self, X, y, now, freq=None):
        if freq is None:
            freq = self.time_loop_freq
        self.split_time_window(now, freq)

        X_train = X.loc[idx[:, self.train_start:self.train_end], :]
        X_test = X.loc[idx[:, self.test_start:self.test_end], :]
        y_train = y.loc[idx[:, self.train_start:self.train_end]]
        y_test = y.loc[idx[:, self.test_start:self.test_end]]

        return X_train, X_test, y_train, y_test

    def split_early_stopping_data(self, X_train, y_train):
        X_early_stopping = X_train.loc[idx[:, self.early_stopping_start:self.early_stopping_end], :]
        y_early_stopping = y_train.loc[idx[:, self.early_stopping_start:self.early_stopping_end]]
        return X_early_stopping, y_early_stopping
