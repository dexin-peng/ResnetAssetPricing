import torch
import time
from torch import optim
from model.model_setup import ModelSetup
from run.dataset import FinancialDataloader
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, config, now, ensemble, X=None, y=None, progress=None, time_index=0, time_total=1, phase_offset=0, progress_total=1):
        self.config = config
        self.now = now
        self.ensemble = ensemble
        self.dataloader = FinancialDataloader(self.config, self.now, X, y)
        self.model = ModelSetup(config).model
        self.optimizer = self._get_optimizer()
        self.writer = SummaryWriter(self.config.io.tensorboard_dir)
        self.epoch_fn = EpochFN(self.config.training.l1, self.config.training.l1_lambda, self.writer, self.now, self.ensemble)
        self.progress = progress
        self.time_index = time_index
        self.time_total = max(time_total, 1)
        self.phase_offset = phase_offset
        self.progress_total = max(progress_total, 1)

    def _get_optimizer(self):
        if self.config.training.l2:
            return optim.AdamW(self.model.parameters(), lr=self.config.training.lr, weight_decay=self.config.training.weight_decay)
        else:
            return optim.Adam(self.model.parameters(), lr=self.config.training.lr)

    def train(self):
        self.train_loader = self.dataloader.train_loader
        self.early_stopping_loader = self.dataloader.early_stopping_loader
        self.test_loader = self.dataloader.test_loader
        self.start_epoch = 0
        self.end_epoch = self.config.training.epochs
        self.previous_time = self.now - self.config.data.time_loop_freq_dt
        if not self.previous_time < self.config.data.time_loop_start:
            if self.config.io.latest_epoch_NE(self.previous_time, self.ensemble) is not None:
                self._load_state(self.previous_time)
        self._train()

    def _train(self):
        self.best_early_stopping_loss = float('inf')
        self.no_improve_epochs = 0
        for epoch in range(self.start_epoch + 1, self.end_epoch + 1):
            self.model.train()
            self.epoch = epoch
            self.epoch_loss = 0
            self._update_progress(None, None, batch_index=0)
            last_progress = time.monotonic()
            for batch_index, (batch_X, batch_y) in enumerate(self.train_loader, start=1):
                loss, _ = self.epoch_fn(True, self.model, self.optimizer, batch_X, batch_y, self.epoch)
                self.epoch_loss += loss.item()
                if time.monotonic() - last_progress >= 1.0:
                    self._update_progress(self.epoch_loss / batch_index, None, batch_index=batch_index)
                    last_progress = time.monotonic()
            avg_loss = self.epoch_loss / len(self.train_loader)
            avg_early_stopping_loss = self._evaluate_early_stopping_loss()
            self.writer.add_scalar(f'{self.now.strftime("%Y-%m-%d")}/ensemble_{self.ensemble}/train/loss', avg_loss, self.epoch)
            self.writer.add_scalar(f'{self.now.strftime("%Y-%m-%d")}/ensemble_{self.ensemble}/early_stop/loss', avg_early_stopping_loss, self.epoch)
            self.config.logger.debug(
                f"Epoch {self.epoch}: Train Loss={avg_loss:.4f}, Early Stop Loss={avg_early_stopping_loss:.4f} "
                f"({self.config.data.early_stopping_start.strftime('%Y-%m-%d')} to {self.config.data.early_stopping_end.strftime('%Y-%m-%d')})"
            )
            self._update_progress(avg_loss, avg_early_stopping_loss)
            if self.epoch % self.config.training.snapshot_freq == 0 or self.epoch == self.end_epoch: self._save_state()
            if self._early_stopping(avg_early_stopping_loss):
                break
        self._evaluate()
        self._complete_progress()

    def _early_stopping(self, avg_early_stopping_loss):
        if avg_early_stopping_loss < self.best_early_stopping_loss:
            self.best_early_stopping_loss = avg_early_stopping_loss
            self.no_improve_epochs = 0
        else:
            self.no_improve_epochs += 1
            self.config.logger.debug(f"Early stop loss did not improve for {self.no_improve_epochs} epoch(s)")

        if self.no_improve_epochs >= self.config.training.early_stopping_patience:
            self._save_state()
            self.config.logger.debug("Early stopping triggered")
            return True
        return False

    def _evaluate_early_stopping_loss(self):
        self.model.eval()
        early_stopping_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in self.early_stopping_loader:
                loss, _ = self.epoch_fn(False, self.model, None, batch_X, batch_y)
                early_stopping_loss += loss.item()
        return early_stopping_loss / len(self.early_stopping_loader)

    def _evaluate(self):
        self.model.eval()
        self.test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in self.test_loader:
                loss, _ = self.epoch_fn(False, self.model, None, batch_X, batch_y)
                self.test_loss += loss.item()
        
        self.avg_test_loss = self.test_loss / len(self.test_loader)
        self.config.logger.debug(f"Test Loss: {self.avg_test_loss:.4f}")
        self.writer.add_scalar(f'{self.now.strftime("%Y-%m-%d")}/ensemble_{self.ensemble}/test/loss', self.avg_test_loss, self.epoch)

    def _save_state(self):
        ckpt_file_path = self.config.io.checkpoint_NE(self.now, self.epoch, self.ensemble)
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch
        }
        torch.save(state_dict, ckpt_file_path)
        if self.config.training.record_grad:
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    self.writer.add_scalar(f'{self.now.strftime("%Y-%m-%d")}/ensemble_{self.ensemble}/grad_norm/{n}', p.grad.detach().cpu().norm(2).item(), self.epoch)

        self.config.logger.debug(f"Saved model to {ckpt_file_path}")

    def _load_state(self, time):
        self.config.logger.debug(f"Loading model from {self.config.io.latest_checkpoint_NE(time, self.ensemble)}")
        ckpt_file_path = self.config.io.latest_checkpoint_NE(time, self.ensemble)
        state_dict = torch.load(ckpt_file_path, map_location=self.config.training.device)
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.epoch = state_dict['epoch']

    def _update_progress(self, avg_loss, avg_early_stopping_loss, batch_index=None):
        if self.progress is None:
            return

        epoch_progress = self.epoch if batch_index is None else self.epoch - 1 + batch_index / max(len(self.train_loader), 1)
        completed = self.phase_offset + self.time_index + (epoch_progress / max(self.end_epoch, 1))
        detail = f"{self.now.strftime('%Y-%m-%d')} epoch {self.epoch:03d}/{self.end_epoch:03d}"
        if batch_index is not None:
            detail += f" batch {batch_index}/{len(self.train_loader)}"
        train_loss = "--" if avg_loss is None else f"{avg_loss:.4f}"
        valid_loss = "--" if avg_early_stopping_loss is None else f"{avg_early_stopping_loss:.4f}"
        self.progress.update(
            phase="train",
            completed=completed,
            total=self.progress_total,
            detail=detail,
            metric_text=f"train loss {train_loss} | valid loss {valid_loss}",
        )

    def _complete_progress(self):
        if self.progress is None:
            return

        self.progress.update(
            phase="train",
            completed=self.phase_offset + self.time_index + 1,
            total=self.progress_total,
            detail=f"{self.now.strftime('%Y-%m-%d')} finished",
            metric_text=f"te {self.avg_test_loss:.4f}",
        )


class EpochFN:
    def __init__(self, l1, l1_lambda, writer, now, ensemble):
        self.l1 = l1
        self.l1_lambda = l1_lambda
        self.writer = writer
        self.now = now
        self.ensemble = ensemble

    def __call__(self, train, model, optimizer, batch_X, batch_y, epoch=None):
        return self.epoch_fn(train, model, optimizer, batch_X, batch_y, epoch)
    
    def l1_norm(self, model):
        """Sum absolute parameter values, excluding names that contain ``bias``.

        See Gu et al. (2020), p. 2235, Eq. (8), for the lasso penalty.
        ``l1_lambda`` scales this sum in the loss.
        """
        return sum(p.abs().sum() for n, p in model.named_parameters() if "bias" not in n)

    def epoch_fn(self, train, model, optimizer, batch_X, batch_y, epoch=None):
        predicted_y = model(batch_X)
        diff = predicted_y - batch_y
        loss = (diff ** 2).mean()

        if self.l1:
            loss = loss + self.l1_lambda * self.l1_norm(model)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss, predicted_y
