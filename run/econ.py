import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import typing
import warnings
import hashlib
import os
import re
from pathlib import Path
from utils.progress_bar import SummaryPrinter
from utils.runtime_device import empty_device_cache


def _realized_return_month_index(index: pd.Index) -> pd.DatetimeIndex:
    """Map a forecast-origin month-end index to the subsequent return month."""
    shifted = pd.DatetimeIndex(index) + pd.offsets.MonthEnd(1)
    shifted.name = index.name
    return shifted


def _with_realized_return_month_index(obj: pd.Series | pd.DataFrame):
    """Return a copy whose month-end index denotes the realized-return month."""
    out = obj.copy()
    out.index = _realized_return_month_index(out.index)
    return out


class EconEvaluator:
    def __init__(self, config):
        self.config = config
        self.summary_printer = SummaryPrinter.from_env()
        self._load_all_predicts()
        self._load_real_ret()
        self._load_rf()
        self._load_mkt_cap()
        self._load_model_params()
        self._load_ff5()

        for group in self.config.econ.econ_mkt_cap:
            if group == 'ALL':
                self.config.logger.debug(f"Economic evaluation for ALL market cap groups...")
                self.calculator = EconCalculator('ALL', self.config, self.all_predicts, self.real, self.rf, self.mkt_cap, self.trainable_params, self.ff5, self.summary_printer)
            elif group == 'TOP1000':
                self.config.logger.debug(f"Economic evaluation for TOP1000 market cap groups...")
                top_mkt_cap_index = self.mkt_cap.groupby('DATE', group_keys=False).apply(lambda s: s.nlargest(1000)).index
                self.calculator = EconCalculator('TOP1000', self.config, self.all_predicts[top_mkt_cap_index], self.real[top_mkt_cap_index], self.rf[top_mkt_cap_index], self.mkt_cap[top_mkt_cap_index], self.trainable_params, self.ff5, self.summary_printer)
            elif group == 'TOP70%':
                self.config.logger.debug(f"Economic evaluation for TOP 70% market cap groups...")
                top_mkt_cap_index = self.mkt_cap.groupby('DATE', group_keys=False).apply(lambda s: s.nlargest(int(len(s) * 0.7))).index
                self.calculator = EconCalculator('TOP70%', self.config, self.all_predicts[top_mkt_cap_index], self.real[top_mkt_cap_index], self.rf[top_mkt_cap_index], self.mkt_cap[top_mkt_cap_index], self.trainable_params, self.ff5, self.summary_printer)
            elif group == 'TOP90%':
                self.config.logger.debug(f"Economic evaluation for Excluding Bottom 10% market cap groups...")
                top_mkt_cap_index = self.mkt_cap.groupby('DATE', group_keys=False).apply(lambda s: s.nlargest(int(len(s) * 0.9))).index
                self.calculator = EconCalculator('TOP90%', self.config, self.all_predicts[top_mkt_cap_index], self.real[top_mkt_cap_index], self.rf[top_mkt_cap_index], self.mkt_cap[top_mkt_cap_index], self.trainable_params, self.ff5, self.summary_printer)
            elif group == 'TOP80%':
                self.config.logger.debug(f"Economic evaluation for Excluding Bottom 20% market cap groups...")
                top_mkt_cap_index = self.mkt_cap.groupby('DATE', group_keys=False).apply(lambda s: s.nlargest(int(len(s) * 0.8))).index
                self.calculator = EconCalculator('TOP80%', self.config, self.all_predicts[top_mkt_cap_index], self.real[top_mkt_cap_index], self.rf[top_mkt_cap_index], self.mkt_cap[top_mkt_cap_index], self.trainable_params, self.ff5, self.summary_printer)
            elif group == 'TOP10%':
                self.config.logger.debug(f"Economic evaluation for Top 10% market cap groups...")
                top_mkt_cap_index = self.mkt_cap.groupby('DATE', group_keys=False).apply(lambda s: s.nlargest(int(len(s) * 0.1))).index
                self.calculator = EconCalculator('TOP10%', self.config, self.all_predicts[top_mkt_cap_index], self.real[top_mkt_cap_index], self.rf[top_mkt_cap_index], self.mkt_cap[top_mkt_cap_index], self.trainable_params, self.ff5, self.summary_printer)
            elif group == 'BOTTOM1000':
                self.config.logger.debug(f"Economic evaluation for BOTTOM1000 market cap groups...")
                bottom_mkt_cap_index = self.mkt_cap.groupby('DATE', group_keys=False).apply(lambda s: s.nsmallest(1000)).index
                self.calculator = EconCalculator('BOTTOM1000', self.config, self.all_predicts[bottom_mkt_cap_index], self.real[bottom_mkt_cap_index], self.rf[bottom_mkt_cap_index], self.mkt_cap[bottom_mkt_cap_index], self.trainable_params, self.ff5, self.summary_printer)
            elif group == 'BOTTOM90%':
                self.config.logger.debug(f"Economic evaluation for Excluding Top 10% market cap groups...")
                bottom_mkt_cap_index = self.mkt_cap.groupby('DATE', group_keys=False).apply(lambda s: s.nsmallest(int(len(s) * 0.9))).index
                self.calculator = EconCalculator('BOTTOM90%', self.config, self.all_predicts[bottom_mkt_cap_index], self.real[bottom_mkt_cap_index], self.rf[bottom_mkt_cap_index], self.mkt_cap[bottom_mkt_cap_index], self.trainable_params, self.ff5, self.summary_printer)
            elif group == 'BOTTOM80%':
                self.config.logger.debug(f"Economic evaluation for Excluding Top 20% market cap groups...")
                bottom_mkt_cap_index = self.mkt_cap.groupby('DATE', group_keys=False).apply(lambda s: s.nsmallest(int(len(s) * 0.8))).index
                self.calculator = EconCalculator('BOTTOM80%', self.config, self.all_predicts[bottom_mkt_cap_index], self.real[bottom_mkt_cap_index], self.rf[bottom_mkt_cap_index], self.mkt_cap[bottom_mkt_cap_index], self.trainable_params, self.ff5, self.summary_printer)
            elif group == 'BOTTOM70%':
                self.config.logger.debug(f"Economic evaluation for Excluding Top 30% market cap groups...")
                bottom_mkt_cap_index = self.mkt_cap.groupby('DATE', group_keys=False).apply(lambda s: s.nsmallest(int(len(s) * 0.7))).index
                self.calculator = EconCalculator('BOTTOM70%', self.config, self.all_predicts[bottom_mkt_cap_index], self.real[bottom_mkt_cap_index], self.rf[bottom_mkt_cap_index], self.mkt_cap[bottom_mkt_cap_index], self.trainable_params, self.ff5, self.summary_printer)
            elif group == 'BOTTOM20%':
                self.config.logger.debug(f"Economic evaluation for Bottom 20% market cap groups...")
                bottom_mkt_cap_index = self.mkt_cap.groupby('DATE', group_keys=False).apply(lambda s: s.nsmallest(int(len(s) * 0.2))).index
                self.calculator = EconCalculator('BOTTOM20%', self.config, self.all_predicts[bottom_mkt_cap_index], self.real[bottom_mkt_cap_index], self.rf[bottom_mkt_cap_index], self.mkt_cap[bottom_mkt_cap_index], self.trainable_params, self.ff5, self.summary_printer)
            elif group == 'BOTTOM10%':
                self.config.logger.debug(f"Economic evaluation for Bottom 10% market cap groups...")
                bottom_mkt_cap_index = self.mkt_cap.groupby('DATE', group_keys=False).apply(lambda s: s.nsmallest(int(len(s) * 0.1))).index
                self.calculator = EconCalculator('BOTTOM10%', self.config, self.all_predicts[bottom_mkt_cap_index], self.real[bottom_mkt_cap_index], self.rf[bottom_mkt_cap_index], self.mkt_cap[bottom_mkt_cap_index], self.trainable_params, self.ff5, self.summary_printer)
            else:
                raise ValueError(f"Invalid market cap group: {group}, please specify from the followings: ALL, TOP1000, TOP70%, TOP90%, TOP80%, TOP10%, BOTTOM1000, BOTTOM70%, BOTTOM90%, BOTTOM10%")
            self.calculator.calculate_all_metrics()
            self.calculator.save()

    def _load_all_predicts(self):
        self.config.logger.debug(f"Loading all predicts...")
        self.all_predicts = []
        for now in pd.date_range(start=self.config.data.time_loop_start, end=self.config.data.time_loop_end, freq=self.config.data.time_loop_freq):
            predicts_to_be_averaged = []
            for ensemble in range(self.config.model.start_ensemble, self.config.model.max_ensemble + 1):
                predicts = pd.read_pickle(self.config.io.predicts_NE(now, ensemble))
                predicts_to_be_averaged.append(predicts)
            predicts = pd.concat(predicts_to_be_averaged, axis=1).mean(axis=1)
            self.all_predicts.append(predicts)
        self.all_predicts = pd.concat(self.all_predicts, axis=0)

    def _load_real_ret(self):
        self.config.logger.debug(f"Loading real returns...")
        self.real = pd.read_pickle(self.config.io.clean_y_path).loc[self.all_predicts.index]

    def _load_rf(self):
        self.config.logger.debug(f"Loading risk-free rates...")
        self.rf = pd.read_pickle(self.config.io.clean_X_path).tbl.loc[self.all_predicts.index]

    def _load_mkt_cap(self):
        self.config.logger.debug(f"Loading lagged market cap...")
        self.mkt_cap = pd.read_pickle(self.config.io.clean_mvel1_path).loc[self.all_predicts.index]

    def _load_model_params(self):
        self.config.logger.debug(f"Loading model parameters...")
        self.latest_ckpt = torch.load(self.config.io.latest_checkpoint_NE(self.config.data.time_loop_end, self.config.model.start_ensemble), map_location=self.config.data.device)
        self.trainable_params = sum(v.numel() for k, v in self.latest_ckpt["model"].items() if k.endswith((".weight", ".bias")))

    def _load_ff5(self):
        self.config.logger.debug(f"Loading FF5 factors...")
        self.ff5 = pd.read_pickle(self.config.io.clean_ff5_path)

class EconCalculator:
    DECILE_FALLBACK_SEED = 42
    DECILE_COUNT = 10

    def __init__(self, prefix, config, all_predicts, real, rf, mkt_cap, trainable_params, ff5, summary_printer=None):
        self.prefix = prefix
        self.config = config
        self.all_predicts = all_predicts
        self.real = real
        self.rf = rf
        self.mkt_cap = mkt_cap
        self.ff5 = ff5
        self.summary_printer = summary_printer
        self.cross_sectionally_all_the_same_predict_flag = False
        assert self.all_predicts.index.equals(self.real.index), "the given predicts have different index from real"
        assert self.all_predicts.index.equals(self.rf.index), "the given predicts have different index from rf"
        assert self.all_predicts.index.equals(self.mkt_cap.index), "the given predicts have different index from mkt_cap"
        self._rename_variables()
        self.trainable_params = trainable_params

        # index level names
        self.date_name = 'DATE'
        names = list(self.all_predicts.index.names)
        if self.date_name not in names:
            raise ValueError("Index must include level 'DATE'")
        self.id_name = next((n for n in names if n != self.date_name), None)
        if self.id_name is None:
            raise ValueError("Index must include a security identifier level besides 'DATE'")

    def _rename_variables(self):
        self.all_predicts.name = 'PredRet'
        self.real.name = 'RealRet'
        self.rf.name = 'RF'
        self.mkt_cap.name = 'MktCap'

    @classmethod
    def _fallback_seed(cls, date_value, group_number: int) -> int:
        payload = f"{cls.DECILE_FALLBACK_SEED}|{date_value}|{group_number}".encode("utf-8")
        return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), byteorder="little")

    @classmethod
    def _seeded_tie_split_decile(cls, s: pd.Series) -> pd.Series:
        out = pd.Series(np.nan, index=s.index, dtype=float)
        clean = s.dropna()
        n_obs = len(clean)
        if n_obs == 0:
            return out

        capacities = np.full(cls.DECILE_COUNT, n_obs // cls.DECILE_COUNT, dtype=int)
        capacities[: n_obs % cls.DECILE_COUNT] += 1
        current_decile = 1
        remaining_capacity = int(capacities[0])

        for group_number, value in enumerate(np.sort(clean.unique())):
            group_index = clean[clean == value].index
            rng = np.random.default_rng(cls._fallback_seed(s.name, group_number))
            shuffled_index = group_index.take(rng.permutation(len(group_index)))
            offset = 0
            while offset < len(shuffled_index):
                while remaining_capacity == 0 and current_decile < cls.DECILE_COUNT:
                    current_decile += 1
                    remaining_capacity = int(capacities[current_decile - 1])
                take = min(remaining_capacity, len(shuffled_index) - offset)
                if take <= 0:
                    break
                out.loc[shuffled_index[offset : offset + take]] = current_decile
                offset += take
                remaining_capacity -= take

        return out

    def _forecast_decile(self, s: pd.Series) -> pd.Series:
        try:
            decile = pd.qcut(
                s,
                self.DECILE_COUNT,
                labels=range(1, self.DECILE_COUNT + 1),
                duplicates='drop',
            )
            return pd.Series(pd.to_numeric(decile, errors="coerce"), index=s.index, dtype=float)
        except ValueError:
            unique_count = s.dropna().nunique()
            self.decile_fallback_months += 1
            if unique_count <= 1:
                self.constant_predict_fallback_months += 1
            self.config.logger.warning(
                f"[{self.prefix}] qcut failed for {s.name}; using seeded tie-split deciles "
                f"(unique forecasts={unique_count}, seed={self.DECILE_FALLBACK_SEED})"
            )
            return self._seeded_tie_split_decile(s)

    def calculate_all_metrics(self):
        self.calculate_r2oos()
        try:
            self.calculate_decile_accuracy()
            self.calculate_decile_return()
            self.calculate_sharpe_ratio()
            self.calculate_ff5_alpha()
            self.calculate_drawdown()
            self.calculate_turnover()
        except Exception as e:
            self.config.logger.warning(f"[{self.prefix}] metrics failed due to {e}, marking as cross_sectionally_all_the_same_predict")
            self.cross_sectionally_all_the_same_predict()

    def cross_sectionally_all_the_same_predict(self):
        self.cross_sectionally_all_the_same_predict_flag = True
        self.decile_accuracy = np.nan
        self.top_bottom_decile_accuracy = np.nan

    def calculate_r2oos(self):
        self.r2oos = 1 - ((self.all_predicts - self.real)**2).sum() / (self.real**2).sum()
        self.real_squared_sum = (self.real**2).sum()
        self.all_predicts_squared_sum = (self.all_predicts**2).sum()

    @staticmethod
    def linear_decile(s: pd.Series) -> pd.Series:
        vmin, vmax = s.min(), s.max()
        if vmax == vmin:
            warnings.warn(f"Unexpected behavior: Every stock's returns in {s.name} are identical; assigning decile rank 5.5 to all of them.", UserWarning)
            return pd.Series(5.5, index=s.index, dtype=float)
        return pd.Series(pd.qcut(s, q=10, labels=False, duplicates='drop'))

    def calculate_decile_accuracy(self):
        self.decile_fallback_months = 0
        self.constant_predict_fallback_months = 0
        self.decile = self.all_predicts.groupby(self.date_name, group_keys=False).apply(self._forecast_decile)
        self.decile = self.decile.reindex(self.all_predicts.index)
        self.decile.name = 'PredRetDecile'
        self.real_decile = self.real.groupby(self.date_name).transform(self.linear_decile).astype(int) + 1
        self.real_decile.name = 'RealRetDecile'
        self.decile_accuracy = (self.real_decile == self.decile).sum() / len(self.decile)
        self.top_bottom_decile_mask = (self.decile == 1) | (self.decile == 10)
        self.top_bottom_decile_accuracy = (self.real_decile[self.top_bottom_decile_mask] == self.decile[self.top_bottom_decile_mask]).sum() / len(self.decile[self.top_bottom_decile_mask])

    def calculate_decile_return(self):
        self.decile_preds_rank_return_rf = pd.concat([self.decile, self.all_predicts, self.real, self.mkt_cap, self.rf], axis=1).reset_index()

        if 'ew' in self.config.econ.econ_weight_type:
            self.ew_ret = self.decile_preds_rank_return_rf.groupby([self.date_name, 'PredRetDecile'], observed=False).RealRet.mean().unstack()
            self.preds_ew_ret = self.decile_preds_rank_return_rf.groupby([self.date_name, 'PredRetDecile'], observed=False).PredRet.mean().groupby('PredRetDecile', observed=False).mean()
            self.ew_ret.name = 'Equal Weighted Decile Return'
            self.preds_ew_ret.name = 'Model Preds Equal Weighted Decile Return'
            self.ew_ret['LS'] = self.ew_ret[10] - self.ew_ret[1]
            self.preds_ew_ret['LS'] = self.preds_ew_ret[10] - self.preds_ew_ret[1]

        if 'vw' in self.config.econ.econ_weight_type:
            self.vw_ret = self.decile_preds_rank_return_rf.groupby([self.date_name, 'PredRetDecile'], observed=False).apply(
                lambda x: ((x.MktCap / x.MktCap.sum()) * x.RealRet).sum(), include_groups=False
            ).unstack()
            self.preds_vw_ret = typing.cast(pd.Series, self.decile_preds_rank_return_rf.groupby([self.date_name, 'PredRetDecile'], observed=False).apply(
                lambda x: ((x.MktCap / x.MktCap.sum()) * x.PredRet).sum(), include_groups=False
            ).groupby('PredRetDecile', observed=False).mean())
            self.vw_ret.name = 'Value Weighted Decile Return'
            self.preds_vw_ret.name = 'Model Preds Value Weighted Decile Return'
            self.vw_ret['LS'] = self.vw_ret[10] - self.vw_ret[1]
            self.preds_vw_ret['LS'] = self.preds_vw_ret[10] - self.preds_vw_ret[1]

    def calculate_sharpe_ratio(self, freq_per_year=12):
        self.continuous_rf_ts = np.log(self.rf.groupby(self.date_name).first() + 1) / freq_per_year
        if 'ew' in self.config.econ.econ_weight_type:
            self.excess_ew = self.ew_ret.subtract(self.continuous_rf_ts, axis=0)
            self.excess_ew.name = 'Equal Weighted Excess Return'
            self.excess_ew_Avg, self.ew_SD, self.excess_ew_SR = self._decile_sharpe_ratio(self.excess_ew, self.ew_ret, freq_per_year)
            self.ew_Avg, self.ew_SD, self.ew_SR = self._decile_sharpe_ratio(self.ew_ret, self.ew_ret, freq_per_year)
        if 'vw' in self.config.econ.econ_weight_type:
            self.excess_vw = self.vw_ret.subtract(self.continuous_rf_ts, axis=0)
            self.excess_vw.name = 'Value Weighted Excess Return'
            self.excess_vw_Avg, self.vw_SD, self.excess_vw_SR = self._decile_sharpe_ratio(self.excess_vw, self.vw_ret, freq_per_year)
            self.vw_Avg, self.vw_SD, self.vw_SR = self._decile_sharpe_ratio(self.vw_ret, self.vw_ret, freq_per_year)

    def _decile_sharpe_ratio(self, ts_deciles_ret_to_be_mean, ts_deciles_ret_to_be_std, freq_per_year):
        avg = ts_deciles_ret_to_be_mean.mean()
        avg.name = 'Avg'
        std = ts_deciles_ret_to_be_std.std()
        std.name = 'SD'
        sr = avg / std * np.sqrt(freq_per_year)
        sr.name = 'Sharpe Ratio'
        return avg, std, sr

    def calculate_ff5_alpha(self, nw_lags: int = 12):
        def _prepare_ff5_decimal(ff5: pd.DataFrame) -> pd.DataFrame:
            fac = ff5[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].copy()
            if fac.abs().values.max() > 1.0:
                fac = fac / 100.0
            fac.columns = ['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA']
            return fac

        def _alpha_t_hac(y: np.ndarray, X: np.ndarray, lags: int = 12) -> typing.Tuple[float, float]:
            T = y.shape[0]
            if T < X.shape[1] + 1:
                return np.nan, np.nan
            XtX = X.T @ X
            XtX_inv = np.linalg.inv(XtX)
            beta = XtX_inv @ (X.T @ y)
            resid = y - X @ beta
            Xu = X * resid[:, None]
            S = Xu.T @ Xu
            for k in range(1, min(lags, T - 1) + 1):
                w = 1.0 - k / (lags + 1.0)
                Xu_f = Xu[k:, :]
                Xu_l = Xu[:-k, :]
                S += w * (Xu_f.T @ Xu_l + Xu_l.T @ Xu_f)
            var_beta = XtX_inv @ S @ XtX_inv
            se = np.sqrt(np.diag(var_beta))
            alpha = beta[0]
            t_alpha = alpha / se[0] if se[0] > 0 else np.nan
            return float(alpha), float(t_alpha)

        def _one_scheme_alpha(ret_ts: pd.Series, fac_df: pd.DataFrame) -> typing.Tuple[float, float]:
            # ret_ts is indexed by forecast origin; factors are indexed by the
            # month in which the subsequent portfolio return is realized.
            realized_ret = _with_realized_return_month_index(ret_ts).rename('LS')
            df = pd.concat([realized_ret, fac_df], axis=1).dropna()
            if df.empty:
                return np.nan, np.nan
            # LS is already the difference between two excess-return deciles,
            # so the risk-free rate cancels and must not be subtracted again.
            y = df['LS'].values.astype(float)
            X = np.column_stack([np.ones(len(df)), df[['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA']].values.astype(float)])
            alpha, t_alpha = _alpha_t_hac(y, X, lags=nw_lags)
            return alpha, t_alpha

        fac_df = _prepare_ff5_decimal(self.ff5)

        if hasattr(self, 'ew_ret'):
            self.ew_ff5_alpha, self.ew_ff5_t = _one_scheme_alpha(self.ew_ret['LS'], fac_df)
        if hasattr(self, 'vw_ret'):
            self.vw_ff5_alpha, self.vw_ff5_t = _one_scheme_alpha(self.vw_ret['LS'], fac_df)

    def calculate_drawdown(self):
        if 'ew' in self.config.econ.econ_weight_type:
            self.ew_drawdown = (self.ew_ret + 1).cumprod().apply(lambda x: (1 - (x / x.cummax())))
            self.ew_drawdown.name = 'Equal Weighted Drawdown'
            self.ew_drawdown_max = - self.ew_drawdown.max()
            self.ew_max_one_month_loss = self.ew_ret.min()
        if 'vw' in self.config.econ.econ_weight_type:
            self.vw_drawdown = (self.vw_ret + 1).cumprod().apply(lambda x: (1 - (x / x.cummax())))
            self.vw_drawdown.name = 'Value Weighted Drawdown'
            self.vw_drawdown_max = - self.vw_drawdown.max()
            self.vw_max_one_month_loss = self.vw_ret.min()

    def calculate_turnover(self):
        r"""
        Compute LS portfolio turnover for configured weighting schemes.
        For each month t -> t+1:
            turnover_{t+1} = sum_i | w_{i,t+1} - w_{i,t} * (1 + r_{i,t+1}) / (1 + sum_j w_{j,t} r_{j,t+1}) |
        LaTeX:
            # \text{Turnover} = \frac{1}{T} \sum_{t=1}^{T} \left(\sum_i \left| w_{i,t+1}
            # - \frac{ w_{i,t}(1 + r_{i,t+1}) }{ 1 + \sum_j w_{j,t} r_{j,t+1} } \right| \right)
        """
        if 'ew' in self.config.econ.econ_weight_type:
            ew_w = self._make_weights_per_date(scheme='ew')
            self.ew_turnover_ts = self._compute_turnover_ts(ew_w)
            self.ew_turnover = float(self.ew_turnover_ts.mean()) if len(self.ew_turnover_ts) else np.nan

        if 'vw' in self.config.econ.econ_weight_type:
            vw_w = self._make_weights_per_date(scheme='vw')
            self.vw_turnover_ts = self._compute_turnover_ts(vw_w)
            self.vw_turnover = float(self.vw_turnover_ts.mean()) if len(self.vw_turnover_ts) else np.nan

    def _make_weights_per_date(self, scheme: str) -> typing.Dict[pd.Timestamp, pd.Series]:
        """
        Construct LS weights per date for a given scheme in {ew, vw}.
        Long leg: decile==10, weights sum to +1; Short leg: decile==1, weights sum to -1.
        Return: dict[date] -> pd.Series indexed by (DATE, ID) with weights.
        """
        weights_by_date: typing.Dict[pd.Timestamp, pd.Series] = {}
        dates = self.all_predicts.index.get_level_values(self.date_name).unique().sort_values()

        # work directly on MultiIndex series for stability
        decile = self.decile
        mktcap = self.mkt_cap

        for d in dates:
            mask_d = decile.index.get_level_values(self.date_name) == d
            dec_d = decile[mask_d]
            if dec_d.empty:
                continue
            idx_long = dec_d[dec_d == 10].index
            idx_short = dec_d[dec_d == 1].index
            if len(idx_long) == 0 or len(idx_short) == 0:
                # skip pathological months
                continue

            if scheme == 'ew':
                w_long = pd.Series(1.0 / len(idx_long), index=idx_long)
                w_short = pd.Series(-1.0 / len(idx_short), index=idx_short)

            elif scheme == 'vw':
                mc_long = mktcap.loc[idx_long]
                mc_short = mktcap.loc[idx_short]
                w_long = mc_long / mc_long.sum()
                w_short = - mc_short / mc_short.sum()

            else:
                raise ValueError(f"Unknown weighting scheme: {scheme}")

            w = pd.concat([w_long, w_short])
            w.name = f'{scheme}_LS_w'
            weights_by_date[d] = w

        return weights_by_date

    def _compute_turnover_ts(self, weights_by_date: typing.Dict[pd.Timestamp, pd.Series]) -> pd.Series:
        """
        Given dict of weights per date (LS), compute monthly turnover time series.
        Its internal index is the forecast origin of the new target weights.
        The saved output maps that origin to the subsequent realized-return month.
        """
        if len(weights_by_date) <= 1:
            return pd.Series(dtype=float, name='Turnover')

        # helper: get returns at a date with ID index only
        def _ret_id_at_date(d: pd.Timestamp) -> pd.Series:
            r = self.real.xs(d, level=self.date_name, drop_level=False)
            # drop DATE level -> index by ID
            r = r.reset_index(level=self.date_name, drop=True)
            return r

        dates_sorted = sorted(weights_by_date.keys())
        out_vals = []
        out_idx = []

        for i in range(len(dates_sorted) - 1):
            d = dates_sorted[i]
            d_next = dates_sorted[i + 1]

            w_t = weights_by_date[d]
            w_tp1_target = weights_by_date[d_next]

            # reduce to ID level for alignment
            w_t_id = w_t.reset_index(level=self.date_name, drop=True)
            w_tp1_id = w_tp1_target.reset_index(level=self.date_name, drop=True)

            r_id = _ret_id_at_date(d)

            # union over IDs
            union_ids = w_t_id.index.union(w_tp1_id.index).union(r_id.index)

            w_t_aligned = w_t_id.reindex(union_ids, fill_value=0.0)
            w_tp1_aligned = w_tp1_id.reindex(union_ids, fill_value=0.0)
            r_aligned = r_id.reindex(union_ids, fill_value=0.0)

            # portfolio simple return for month d -> d+1
            rp = float((w_t_aligned * r_aligned).sum())
            denom = 1.0 + rp

            if np.isclose(denom, 0.0):
                turnover = np.nan
            else:
                drift = w_t_aligned * (1.0 + r_aligned) / denom
                turnover = float((w_tp1_aligned - drift).abs().sum())

            out_vals.append(turnover)
            out_idx.append(d_next)

        turnover_ts = pd.Series(out_vals, index=pd.Index(out_idx, name=self.date_name), name='Turnover')
        return turnover_ts

    def save_decile_return(self):
        if 'ew' in self.config.econ.econ_weight_type:
            _with_realized_return_month_index(self.ew_ret).to_csv(
                self.config.io.econ_ew_decile_return_path(self.prefix)
            )
        if 'vw' in self.config.econ.econ_weight_type:
            _with_realized_return_month_index(self.vw_ret).to_csv(
                self.config.io.econ_vw_decile_return_path(self.prefix)
            )

    def save_turnover(self):
        if 'ew' in self.config.econ.econ_weight_type and hasattr(self, 'ew_turnover_ts'):
            _with_realized_return_month_index(self.ew_turnover_ts).to_frame().to_csv(
                self.config.io.econ_ew_turnover_path(self.prefix)
            )
        if 'vw' in self.config.econ.econ_weight_type and hasattr(self, 'vw_turnover_ts'):
            _with_realized_return_month_index(self.vw_turnover_ts).to_frame().to_csv(
                self.config.io.econ_vw_turnover_path(self.prefix)
            )

    def save(self):
        if not self.cross_sectionally_all_the_same_predict_flag:
            self.save_decile_return()
            self.save_turnover()
        self.brief_results = {
            'Asset Prefix': self.config.io.asset_prefix,
            'Model Name': self.config.model.name,
            'Model Type': self.config.model.type,
            'Layers Depth': len(self.config.model.hidden_sizes),
            'Decile Accuracy': f"{self.decile_accuracy * 100}%",
            'Top Bottom Decile Accuracy': f"{self.top_bottom_decile_accuracy * 100}%",
            'Out-of-Sample R2': f"{self.r2oos * 100}%",
            'Trainable Parameters': self.trainable_params,
            'Trainable Parameters Magnitude': f"10^{int(np.floor(np.log10(self.trainable_params))) if self.trainable_params > 0 else 0}",
            'Hidden Sizes': self.config.model.hidden_sizes,
            'Squared Sum of Real Returns': self.real_squared_sum,
            'Squared Sum of All Predicted Returns': self.all_predicts_squared_sum,
            'Seeded Tie-Split Decile Fallback Months': getattr(self, 'decile_fallback_months', 0),
            'Constant-Predict Decile Fallback Months': getattr(self, 'constant_predict_fallback_months', 0),
        }
        if not self.cross_sectionally_all_the_same_predict_flag:
            if 'vw' in self.config.econ.econ_weight_type:
                self.brief_results['Value Weighted Long Short Sharpe Ratio'] = self.vw_SR['LS']
                self.brief_results['Value Weighted Long Short Excess Sharpe Ratio'] = self.excess_vw_SR['LS']
                self.brief_results['Value Weighted Long Short Maximum Drawdown'] = f"{self.vw_drawdown_max.LS * 100}%"
                self.brief_results['Value Weighted Long Short Maximum One Month Loss'] = f"{self.vw_max_one_month_loss.LS * 100}%"
                self.brief_results['Monthly Value Weighted Long Short Preds LS'] = f"{self.preds_vw_ret['LS'] * 100}%"
                if hasattr(self, 'vw_ff5_alpha'):
                    self.brief_results['Value Weighted LS FF5 Alpha/%'] = self.vw_ff5_alpha * 100
                    self.brief_results['Value Weighted LS FF5 t-stat'] = self.vw_ff5_t
                if hasattr(self, 'vw_turnover'):
                    self.brief_results['Value Weighted LS Turnover/%'] = self.vw_turnover * 100

                self.vw_decile_SR_table = pd.concat([self.preds_vw_ret * 100, self.vw_Avg * 100, self.vw_SD * 100, self.vw_SR], axis=1)
                self.vw_decile_SR_table.columns = ['Monthly Raw Preds Mean/%', 'Monthly Excess Mean/%', 'Monthly Std*100', 'Sharpe Ratio']
                self.vw_decile_SR_table.to_csv(self.config.io.econ_vw_decile_SR_table_path(self.prefix))

            if 'ew' in self.config.econ.econ_weight_type:
                self.brief_results['Equal Weighted Long Short Sharpe Ratio'] = self.ew_SR['LS']
                self.brief_results['Equal Weighted Long Short Excess Sharpe Ratio'] = self.excess_ew_SR['LS']
                self.brief_results['Equal Weighted Long Short Maximum Drawdown'] = f"{self.ew_drawdown_max.LS * 100}%"
                self.brief_results['Equal Weighted Long Short Maximum One Month Loss'] = f"{self.ew_max_one_month_loss.LS * 100}%"
                self.brief_results['Monthly Equal Weighted Long Short Preds LS'] = f"{self.preds_ew_ret['LS'] * 100}%"
                if hasattr(self, 'ew_ff5_alpha'):
                    self.brief_results['Equal Weighted LS FF5 Alpha/%'] = self.ew_ff5_alpha * 100
                    self.brief_results['Equal Weighted LS FF5 t-stat'] = self.ew_ff5_t
                if hasattr(self, 'ew_turnover'):
                    self.brief_results['Equal Weighted LS Turnover/%'] = self.ew_turnover * 100

                self.ew_decile_SR_table = pd.concat([self.preds_ew_ret * 100, self.ew_Avg * 100, self.ew_SD * 100, self.ew_SR], axis=1)
                self.ew_decile_SR_table.columns = ['Monthly Raw Preds Mean/%', 'Monthly Excess Mean/%', 'Monthly Std*100', 'Sharpe Ratio']
                self.ew_decile_SR_table.to_csv(self.config.io.econ_ew_decile_SR_table_path(self.prefix))

        self.brief_results = pd.Series(self.brief_results, name=self.prefix)
        self.brief_results.index.name = 'Brief Results'
        self.config.logger.debug(f"Brief Results For {self.prefix}: {self.brief_results.to_string()}")
        self.brief_results.to_csv(self.config.io.econ_brief_results_path(self.prefix))
        self._emit_terminal_summary()

    def _emit_terminal_summary(self):
        if self.summary_printer is None:
            return

        sr_map = {}
        if hasattr(self, 'ew_SR'):
            sr_map['ew'] = self.ew_SR['LS']
        if hasattr(self, 'vw_SR'):
            sr_map['vw'] = self.vw_SR['LS']

        self.summary_printer.emit_econ_result(
            asset_prefix=self.config.io.asset_prefix,
            group=self.prefix,
            sr_map=sr_map,
        )


class GroupAblationEvaluator:
    """Save portfolio results and stock forecasts after zeroing each JKP cluster."""

    CLUSTER_LABELS_PATH = Path(__file__).resolve().parents[1] / "source_data" / "cluster_labels.csv"
    OUTPUT_ROOT = "ALL/group_ablation"
    STOCK_PREDICTIONS_FILE = "stock_predictions.pkl.gz"
    EXPECTED_CLUSTER_COUNT = 13

    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config.training.device)
        self.summary_printer = SummaryPrinter.from_env()
        self.clusters = self._load_clusters()
        self.pending_clusters = [
            cluster for cluster in self.clusters if not self._output_complete(cluster)
        ]
        if not self.pending_clusters:
            self.config.logger.info("All group-ablation econ and stock-prediction outputs already exist; nothing to run.")
            return

        self.time_points = list(
            pd.date_range(
                start=self.config.data.time_loop_start,
                end=self.config.data.time_loop_end,
                freq=self.config.data.time_loop_freq,
            )
        )
        self.ensembles = list(
            range(self.config.model.start_ensemble, self.config.model.max_ensemble + 1)
        )
        if not self.ensembles:
            raise ValueError("Group ablation requires at least one ensemble member.")

        self.checkpoints = self._preflight_inputs()
        self.columns, self.mask_indices = self._load_and_validate_columns()
        self.model = self._build_model()
        predictions, index = self._predict_pending_clusters()
        del self.model
        empty_device_cache(str(self.device))
        gpu_done_fd = os.environ.get("APRNetRewrite_GPU_DONE_FD")
        if gpu_done_fd is not None:
            os.write(int(gpu_done_fd), b"1")
            os.close(int(gpu_done_fd))
        self._evaluate_and_save(predictions, index)

    @staticmethod
    def _cluster_slug(cluster: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", cluster.lower()).strip("_")

    def _output_prefix(self, cluster: str) -> str:
        return f"{self.OUTPUT_ROOT}/{self._cluster_slug(cluster)}"

    def _output_complete(self, cluster: str) -> bool:
        directory = Path(self.config.io.econ_dir) / self._output_prefix(cluster)
        # Legacy econ-only results must rerun inference to recover stock ranks.
        return all(
            path.is_file() and path.stat().st_size > 0
            for path in (directory / "brief_results.csv", directory / self.STOCK_PREDICTIONS_FILE)
        )

    def _save_stock_predictions(
        self, cluster: str, predictions: pd.Series, real: pd.Series
    ) -> None:
        if not predictions.index.equals(real.index):
            raise ValueError("Group-ablation forecasts and realized returns must share an index.")
        # Keep raw forecasts, including ties, and the original stock/formation-month
        # index. Unlike econ return tables, DATE is not shifted to the return month.
        frame = pd.DataFrame({"PredRet": predictions, "RealRet": real})
        frame.attrs.update(
            model=self.config.model.name,
            theme=cluster,
            start_ensemble=self.config.model.start_ensemble,
            max_ensemble=self.config.model.max_ensemble,
            date_convention="DATE is the forecast-origin month; RealRet is the next-month return.",
        )
        path = Path(self.config.io.econ_dir) / self._output_prefix(cluster) / self.STOCK_PREDICTIONS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            frame.to_pickle(temporary, compression={"method": "gzip", "compresslevel": 1})
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _load_clusters(self) -> dict[str, tuple[str, ...]]:
        if not self.CLUSTER_LABELS_PATH.is_file():
            raise FileNotFoundError(
                f"Missing JKP cluster labels: {self.CLUSTER_LABELS_PATH}"
            )
        labels = pd.read_csv(self.CLUSTER_LABELS_PATH)
        required = {"characteristic", "cluster"}
        if set(labels.columns) != required:
            raise ValueError(
                f"Expected cluster label columns {sorted(required)}, got {labels.columns.tolist()}."
            )
        if len(labels) != self.config.data.number_of_chars:
            raise ValueError(
                f"Expected {self.config.data.number_of_chars} cluster labels, found {len(labels)}."
            )
        if labels.isna().any().any() or labels["characteristic"].duplicated().any():
            raise ValueError("Cluster labels must be complete and unique by characteristic.")

        grouped = {
            str(cluster): tuple(sorted(part["characteristic"].astype(str)))
            for cluster, part in labels.groupby("cluster", sort=True)
        }
        if len(grouped) != self.EXPECTED_CLUSTER_COUNT:
            raise ValueError(
                f"Expected {self.EXPECTED_CLUSTER_COUNT} JKP clusters, found {len(grouped)}."
            )
        slugs = [self._cluster_slug(cluster) for cluster in grouped]
        if len(slugs) != len(set(slugs)):
            raise ValueError("JKP cluster names produce duplicate output directory names.")
        return grouped

    def _preflight_inputs(self) -> dict[tuple[pd.Timestamp, int], Path]:
        required_paths = [
            Path(self.config.io.clean_X_path),
            Path(self.config.io.clean_y_path),
            Path(self.config.io.clean_mvel1_path),
            Path(self.config.io.clean_ff5_path),
        ]
        checkpoints: dict[tuple[pd.Timestamp, int], Path] = {}
        missing: list[str] = [str(path) for path in required_paths if not path.is_file()]

        for now in self.time_points:
            for path in (
                Path(self.config.io.X_test_tensor_path(now)),
                Path(self.config.io.X_test_index_path(now)),
                Path(self.config.io.X_columns_path(now)),
            ):
                if not path.is_file():
                    missing.append(str(path))
            for ensemble in self.ensembles:
                checkpoint = self.config.io.latest_checkpoint_NE(now, ensemble)
                if checkpoint is None or not Path(checkpoint).is_file():
                    missing.append(
                        str(Path(self.config.io.checkpoint_dir_NE(now, ensemble)) / "epoch_*.pth")
                    )
                else:
                    checkpoints[(now, ensemble)] = Path(checkpoint)

        if missing:
            preview = "\n".join(f"- {path}" for path in missing[:20])
            remainder = len(missing) - min(len(missing), 20)
            suffix = f"\n- ... and {remainder} more" if remainder else ""
            raise FileNotFoundError(
                "Group ablation requires complete clean caches, pre-split test data, and checkpoints:\n"
                f"{preview}{suffix}"
            )
        return checkpoints

    def _load_and_validate_columns(self) -> tuple[pd.Index, dict[str, tuple[int, ...]]]:
        expected_columns: pd.Index | None = None
        for now in self.time_points:
            columns = pd.Index(pd.read_pickle(self.config.io.X_columns_path(now))).astype(str)
            if expected_columns is None:
                expected_columns = columns
            elif not columns.equals(expected_columns):
                raise ValueError(f"Feature columns changed at {now.strftime('%Y-%m-%d')}.")

        if expected_columns is None:
            raise ValueError("No test-period feature columns were found.")
        expected_total = self.config.data.number_of_chars + self.config.data.number_of_macro
        if len(expected_columns) != expected_total or expected_columns.duplicated().any():
            raise ValueError(
                f"Expected {expected_total} unique model inputs, got {len(expected_columns)}."
            )

        macro_columns = set(map(str, self.config.data.selected_macro))
        missing_macro = sorted(macro_columns.difference(expected_columns))
        if missing_macro:
            raise ValueError(f"Missing configured macro predictors: {missing_macro}")
        firm_columns = set(expected_columns).difference(macro_columns)
        mapped_columns = {
            characteristic
            for characteristics in self.clusters.values()
            for characteristic in characteristics
        }
        if firm_columns != mapped_columns:
            missing = sorted(firm_columns.difference(mapped_columns))
            extra = sorted(mapped_columns.difference(firm_columns))
            raise ValueError(
                f"Cluster labels do not match model firm characteristics; missing={missing}, extra={extra}."
            )

        masks = {
            cluster: tuple(
                int(expected_columns.get_loc(characteristic))
                for characteristic in self.clusters[cluster]
            )
            for cluster in self.pending_clusters
        }
        return expected_columns, masks

    def _build_model(self):
        from model.model_setup import ModelSetup

        model = ModelSetup(self.config).model
        model.eval()
        return model

    def _predict_pending_clusters(self) -> tuple[np.ndarray, pd.Index]:
        cluster_count = len(self.pending_clusters)
        ensemble_count = len(self.ensembles)
        prediction_blocks: list[np.ndarray] = []
        index_blocks: list[pd.Index] = []
        device_masks = [
            torch.as_tensor(self.mask_indices[cluster], device=self.device, dtype=torch.long)
            for cluster in self.pending_clusters
        ]

        for now in self.time_points:
            tensor_path = self.config.io.X_test_tensor_path(now)
            index_path = self.config.io.X_test_index_path(now)
            X_test = torch.load(tensor_path, map_location=self.device)
            index = pd.read_pickle(index_path)
            if X_test.ndim != 2 or X_test.shape != (len(index), len(self.columns)):
                raise ValueError(
                    f"Unexpected test tensor shape at {now.strftime('%Y-%m-%d')}: "
                    f"tensor={tuple(X_test.shape)}, index={len(index)}, columns={len(self.columns)}."
                )

            prediction_sum = torch.zeros(
                (len(index), cluster_count), device=self.device, dtype=torch.float32
            )
            for ensemble in self.ensembles:
                checkpoint = torch.load(
                    self.checkpoints[(now, ensemble)], map_location=self.device
                )
                self.model.load_state_dict(checkpoint["model"])
                self.model.eval()
                with torch.inference_mode():
                    for start in range(0, len(index), self.config.training.batch_size):
                        stop = min(start + self.config.training.batch_size, len(index))
                        batch = X_test[start:stop]
                        masked = batch.unsqueeze(0).expand(cluster_count, -1, -1).clone()
                        for cluster_index, mask in enumerate(device_masks):
                            masked[cluster_index].index_fill_(1, mask, 0.0)
                        predicted = self.model(
                            masked.reshape(cluster_count * len(batch), len(self.columns))
                        ).reshape(cluster_count, len(batch)).transpose(0, 1)
                        prediction_sum[start:stop] += predicted
                self.config.logger.info(
                    f"Group ablation predicted {now.strftime('%Y-%m-%d')} "
                    f"ensemble {ensemble - self.ensembles[0] + 1}/{ensemble_count} "
                    f"for {self.config.model.name}"
                )

            prediction_sum = (prediction_sum / float(ensemble_count)).cpu().numpy()
            if not np.isfinite(prediction_sum).all():
                raise ValueError(f"Non-finite group-ablation predictions at {now.strftime('%Y-%m-%d')}.")
            prediction_blocks.append(prediction_sum)
            index_blocks.append(index)
            del X_test, prediction_sum
            empty_device_cache(str(self.device))

        predictions = np.concatenate(prediction_blocks, axis=0)
        index = index_blocks[0].append(index_blocks[1:])
        if len(index) != len(predictions) or index.duplicated().any():
            raise ValueError("Group-ablation prediction index is incomplete or duplicated.")
        return predictions, index

    def _evaluate_and_save(self, predictions: np.ndarray, index: pd.Index) -> None:
        real = pd.read_pickle(self.config.io.clean_y_path).loc[index]
        rf = pd.read_pickle(self.config.io.clean_X_path)["tbl"].loc[index]
        mkt_cap = pd.read_pickle(self.config.io.clean_mvel1_path).loc[index]
        ff5 = pd.read_pickle(self.config.io.clean_ff5_path)
        final_checkpoint = torch.load(
            self.checkpoints[(self.time_points[-1], self.ensembles[0])], map_location="cpu"
        )
        trainable_params = sum(
            value.numel()
            for name, value in final_checkpoint["model"].items()
            if name.endswith((".weight", ".bias"))
        )

        for cluster_index, cluster in enumerate(self.pending_clusters):
            prefix = self._output_prefix(cluster)
            cluster_predictions = pd.Series(
                predictions[:, cluster_index], index=index, name=self.config.model.name
            )
            calculator = EconCalculator(
                prefix,
                self.config,
                cluster_predictions,
                real,
                rf,
                mkt_cap,
                trainable_params,
                ff5,
                self.summary_printer,
            )
            calculator.calculate_all_metrics()
            calculator.save()
            # Publish this file last so an interrupted legacy-result backfill
            # cannot be mistaken for a completed theme on the next run.
            self._save_stock_predictions(cluster, cluster_predictions, real)
            self.config.logger.info(
                f"Saved {cluster} group-ablation econ outputs and stock predictions to "
                f"{Path(self.config.io.econ_dir) / prefix}"
            )
