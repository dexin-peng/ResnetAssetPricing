import pandas as pd
import numpy as np


class PreprocessHandler:
    def __init__(self, price_characteristics):
        self.price_characteristics = price_characteristics

    @property
    def fill_missing_value_with_median(self):
        missing_count = self.price_characteristics.isna().sum().sum()
        self.price_characteristics.fillna(self.price_characteristics.groupby('DATE').transform('median'),inplace=True)
        missing_count_after_fill_median = self.price_characteristics.isna().sum().sum()
        print(f"Characteristics filled with median count: {missing_count - missing_count_after_fill_median}, percentage: {(missing_count - missing_count_after_fill_median)/self.price_characteristics.size*100:.2f}%")
        print(f"Cross-sectionally all missing characteristics: {missing_count_after_fill_median}, percentage: {missing_count_after_fill_median/self.price_characteristics.size*100:.2f}%")

    @property
    def fill_missing_value_with_0(self):
        missing_count = self.price_characteristics.isna().sum().sum()
        self.price_characteristics.fillna(0, inplace=True)
        print(f"Characteristics filled with 0 count: {missing_count}, percentage: {missing_count/self.price_characteristics.size*100:.2f}%")

    @staticmethod
    def comprehensive_data_check(X_train, X_test, y_train, y_test, config, now):
        print(f"Checking: Now: {now.strftime('%Y-%m-%d')}, Train: {config.data.train_start.strftime('%Y-%m-%d')} to {config.data.train_end.strftime('%Y-%m-%d')}, Test: {config.data.test_start.strftime('%Y-%m-%d')} to {config.data.test_end.strftime('%Y-%m-%d')}")
        assert X_train.shape[1] == config.data.number_of_chars + config.data.number_of_macro, 'The shape[1] of training data is not correct'
        assert X_test.shape[1] == config.data.number_of_chars + config.data.number_of_macro, 'The shape[1] of test data is not correct'
        assert X_train.shape[1] == X_test.shape[1], 'The shape[1] of training and test data should be the same'
        assert X_train.shape[0] == y_train.shape[0], 'The shape of y should be the same as the shape[0] of training data'
        assert X_test.shape[0] == y_test.shape[0], 'The shape of y should be the same as the shape[0] of test data'
        assert X_train.index.equals(y_train.index), 'The index of training data is not correct'
        assert X_test.index.equals(y_test.index), 'The index of test data is not correct'
        assert len(X_train) != 0, 'The length of training data is 0'
        assert len(X_test) != 0, 'The length of test data is 0'
        assert (np.dtype('float64') == X_train.dtypes.values).all(), 'Unexpected datatypes in training data'
        assert (np.dtype('float64') == X_test.dtypes.values).all(), 'Unexpected datatypes in test data'
        assert np.dtype('float64') == y_train.dtypes, 'Unexpected datatypes in training data'
        assert np.dtype('float64') == y_test.dtypes, 'Unexpected datatypes in test data'
        assert pd.isna(X_train).sum().sum() == 0, 'There are missing values in training data'
        assert pd.isna(X_test).sum().sum() == 0, 'There are missing values in test data'
        assert pd.isna(y_train).sum() == 0, 'There are missing values in training data'
        assert pd.isna(y_test).sum() == 0, 'There are missing values in test data'
        assert np.isinf(X_train).sum().sum() == 0, 'There are infinite values in training data'
        assert np.isinf(X_test).sum().sum() == 0, 'There are infinite values in test data'
        assert np.isinf(y_train).sum() == 0, 'There are infinite values in training data'
        assert np.isinf(y_test).sum() == 0, 'There are infinite values in test data'
        assert X_train.index.duplicated().sum() == 0, 'There are duplicate datetime-stkcd pairs in training data'
        assert X_test.index.duplicated().sum() == 0, 'There are duplicate datetime-stkcd pairs in test data'
        assert X_train.drop(columns=config.data.selected_macro).max().max() <= 1, 'There are values greater than 1 in training data'
        assert X_test.drop(columns=config.data.selected_macro).max().max() <= 1, 'There are values greater than 1 in test data'
        assert X_train.drop(columns=config.data.selected_macro).min().min() >= -1, 'There are values less than -1 in training data'
        assert X_test.drop(columns=config.data.selected_macro).min().min() >= -1, 'There are values less than -1 in test data'
        print(f"Pass")
