import torch
import numpy as np
import pandas as pd
from model.model_setup import ModelSetup


class Predictor():
    def __init__(self, config, now, ensemble, progress=None, time_index=0, time_total=1, phase_offset=0, progress_total=1):
        self.now = now
        self.config = config
        self.ensemble = ensemble
        self.model = ModelSetup(self.config).model
        self.progress = progress
        self.time_index = time_index
        self.time_total = max(time_total, 1)
        self.phase_offset = phase_offset
        self.progress_total = max(progress_total, 1)

    def predict(self, dataloader):
        self.dataloader = dataloader
        self.test_loader = self.dataloader.test_loader
        self._load_state(self.now)
        self._predict(self.now)

    def _predict(self, now):
        self.model.eval()
        self.val_loss = 0.0
        self.test_loss = 0.0
        total_batches = max(len(self.test_loader), 1)
        progress_interval = max(1, total_batches // 5)
        with torch.no_grad():
            self.predicts = []
            for batch_idx, (batch_X, batch_y) in enumerate(self.test_loader, start=1):
                predicted_y = self.model(batch_X)
                self.predicts.append(predicted_y.cpu().numpy())
                if self.progress is not None and (batch_idx == 1 or batch_idx == total_batches or batch_idx % progress_interval == 0):
                    self.progress.update(
                        phase="predict",
                        completed=self.phase_offset + self.time_index + (batch_idx / total_batches),
                        total=self.progress_total,
                        detail=f"{self.now.strftime('%Y-%m-%d')} batch {batch_idx:02d}/{total_batches:02d}",
                        metric_text=self._predict_metric_text(),
                    )
            preds_array = np.concatenate(self.predicts, axis=0)
            self.predicts = pd.Series(preds_array.ravel(), index=self.dataloader.y_test_index)
        self.predicts.to_pickle(self.config.io.predicts_NE(now, self.ensemble))
        if self.progress is not None:
            self.progress.update(
                phase="predict",
                completed=self.phase_offset + self.time_index + 1,
                total=self.progress_total,
                detail=f"{self.now.strftime('%Y-%m-%d')} saved",
                metric_text=f"rows {len(self.predicts):,}",
            )

    def _load_state(self, time):
        self.config.logger.debug(f"Loading model from {self.config.io.latest_checkpoint_NE(time, self.ensemble)}")
        ckpt_file_path = self.config.io.latest_checkpoint_NE(time, self.ensemble)
        state_dict = torch.load(ckpt_file_path, map_location=self.config.data.device)
        self.model.load_state_dict(state_dict['model'])
        self.epoch = state_dict['epoch']

    def _predict_metric_text(self):
        return "pred"
