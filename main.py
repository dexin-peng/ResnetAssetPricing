import os
import random
import sys
import numpy as np
import pandas as pd
import torch
import argparse

from run.dataset import FinancialDataloader
from utils.progress_bar import JobProgress
from utils.runtime_device import default_device, resolve_device, empty_device_cache
from config.register import PUBLIC_MODEL_CONFIG_CHOICES, resolve_config_cls


parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true", default=False, help="train the model")
parser.add_argument("--predict", action="store_true", default=False, help="use the pre-trained model to predict")
parser.add_argument("--econ", action="store_true", default=False, help="economic evaluation, if set train and predict to True, econ will be overridden by False. Due to the results are stochastic, economic evaluation should be run AFTER ALL ENSEMBLES are trained and predicted")
parser.add_argument("--group_ablation", action="store_true", default=False, help="run test-set inference after zeroing each JKP characteristic cluster and save ALL-stock econ outputs")
parser.add_argument(
    "--config_cls",
    type=str,
    default="ResNetpS4D8Config",
    help=f"config class. Public names: {', '.join(PUBLIC_MODEL_CONFIG_CHOICES)}",
)
parser.add_argument("--epochs", type=int, default=100, help="max training epochs")
parser.add_argument("--record_grad", action="store_true", default=False, help="record the training gradient to tensorboard or not. It is enabled only if you want to inspect the gradient of the model. You should have at least 350 GiB free disk space before enabling this setting, and disk space it costs depends on the model size and the number of epochs.")
parser.add_argument("--batch_size", type=int, default=10000, help="training and testing share the same batch size")
parser.add_argument("--early_stopping_patience", type=int, default=5, help="early stopping patience. The training will stop if the early stop loss on the latest five training years does not improve for `early_stopping_patience` epochs")
parser.add_argument("--l1", action="store_true", default=False, help="add L1 regularization to the loss, excluding parameters whose names contain bias")
parser.add_argument("--l1_lambda", type=float, default=None, help="L1 regularization strength")
parser.add_argument("--l2", action="store_true", default=False, help="enable decoupled weight decay with AdamW; Adam is used when this flag is omitted")
parser.add_argument("--weight_decay", type=float, default=None, help="decoupled weight decay coefficient used with --l2")
parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "Adam", "ADAM"], help="Adam optimizer family; --l2 selects AdamW with decoupled weight decay")
parser.add_argument("--initial_lr", type=float, default=None, help="initial learning rate")
parser.add_argument("--asset_prefix", type=str, default='default', help="asset prefix, e.g. default. Then you can find all the results in ./asset/default")
parser.add_argument("--ensemble", type=int, default=0, help="ensemble number 0-9, this name is in consistent with Gu et al. (2020), because the results are stochastic, run them 10 times and average the prediction results. In this code repo, random seed is linked with ensemble number. To most efficiently utilize multi process training, it's recommended to have at least 30GB GPU memory, otherwise you may experience out of memory error")
parser.add_argument("--start_ensemble", type=int, default=0, help="Econ evaluation will start from this ensemble number")
parser.add_argument("--max_ensemble", type=int, default=9, help="Econ evaluation will end at this ensemble number")
parser.add_argument("--device", type=str, default=default_device(), help="device: auto, cuda:N, mps or cpu; default selects CUDA, Apple MPS, then CPU")
parser.add_argument("--init_pre_split_dataset", action="store_true", default=False, help="initialize the pre split dataset to greatly increse the training speed by reducing the time cost of data loading and copying between CPU and GPU. If set, the dataset will pre split and save to the tmp directory. NOTE, if you want to use the pre split dataset, you are required to have at least 50GB free disk space")
parser.add_argument("--use_pre_split_dataset", action="store_true", default=False, help="load cached annual data splits from tmp; omit this flag to disable")
parser.add_argument("--econ_mkt_cap", type=str, nargs='+', default=['ALL', 'BOTTOM80%', 'TOP80%'], choices=['ALL', 'TOP1000', 'TOP70%', 'TOP90%', 'TOP80%', 'BOTTOM1000', 'BOTTOM70%', 'BOTTOM80%', 'BOTTOM90%', 'BOTTOM10%'], help="which market cap group(s) to do economic evaluation, can be a list, e.g. --econ_mkt_cap ALL TOP90%% TOP80%%")
parser.add_argument("--econ_weight_type", type=str, nargs='+', default=['vw'], choices=['ew', 'vw'], help="which weight type to do economic evaluation, can be a list, e.g. --econ_weight_type ew vw")


if __name__ == "__main__":

    args = parser.parse_args()
    if args.group_ablation and not args.use_pre_split_dataset:
        parser.error("--group_ablation requires --use_pre_split_dataset")
    try:
        args.device = resolve_device(args.device)
    except ValueError as exc:
        parser.error(str(exc))
    ensemble = args.ensemble
    try:
        config_cls = resolve_config_cls(args.config_cls)
    except KeyError as exc:
        parser.error(
            f"Unknown config class '{args.config_cls}'. Public names: {', '.join(PUBLIC_MODEL_CONFIG_CHOICES)}"
        )
        raise exc

    config = config_cls(args)
    time_points = list(pd.date_range(start=config.data.time_loop_start, end=config.data.time_loop_end, freq=config.data.time_loop_freq))
    active_phases = [phase for phase in ("train", "predict") if getattr(args, phase)]
    progress_total = max(len(active_phases) * max(len(time_points), 1), 1)
    phase_offsets = {phase: index * len(time_points) for index, phase in enumerate(active_phases)}
    progress = JobProgress.from_env(fallback_label=f"{config.model.name} e{ensemble:02d}")

    if args.init_pre_split_dataset:
        config.logger.info(f"When initializing the pre split dataset. `train`, `predict`, `econ`, `group_ablation`, `use_pre_split_dataset` are overridden by False")
        args.train = False
        args.predict = False
        args.econ = False
        args.group_ablation = False
        args.use_pre_split_dataset = False

    if args.train and args.predict and args.econ:
        config.logger.info(f"Run economic evaluation AFTER ALL ENSEMBLES are trained and predicted. This setting is overridden by False")
        args.econ = False

    config.logger.debug(f"args: {args}")

    if not args.use_pre_split_dataset:
        X = pd.read_pickle(config.io.clean_X_path)
        y = pd.read_pickle(config.io.clean_y_path)
    else:
        X = None
        y = None

    if args.init_pre_split_dataset:
        for now in time_points:
            dataloader = FinancialDataloader(config, now, X, y)
            del dataloader
            empty_device_cache(args.device)

    if args.train:
        from run.trainer import Trainer
        for time_index, now in enumerate(time_points):
            random.seed(ensemble)
            np.random.seed(ensemble)
            torch.manual_seed(ensemble)
            if torch.device(args.device).type == "cuda":
                torch.cuda.manual_seed(ensemble)
                torch.cuda.manual_seed_all(ensemble)
                torch.backends.cudnn.deterministic = True
            if config.training.continue_training:
                if config.io.latest_checkpoint_NE(now, ensemble) is not None:
                    config.logger.debug(f"Skipping training for now={now.strftime('%Y-%m-%d')}, model={config.model.name}, ensemble={ensemble} because it has been trained")
                    progress.mark_skipped(
                        completed=phase_offsets["train"] + time_index + 1,
                        total=progress_total,
                        detail=f"{now.strftime('%Y-%m-%d')} checkpoint exists",
                    )
                    continue
            trainer = Trainer(
                config,
                now,
                ensemble,
                X,
                y,
                progress=progress,
                time_index=time_index,
                time_total=len(time_points),
                phase_offset=phase_offsets["train"],
                progress_total=progress_total,
            )
            trainer.train()
            del trainer
            empty_device_cache(args.device)

    if args.predict:
        from run.predictor import Predictor
        for time_index, now in enumerate(time_points):
            config.logger.debug(f"Predicting for now={now.strftime('%Y-%m-%d')}, model={config.model.name}, ensemble={ensemble}")
            if os.path.exists(config.io.predicts_NE(now, ensemble)):
                config.logger.debug(f"Skip now={now.strftime('%Y-%m-%d')}, model={config.model.name}, ensemble={ensemble}")
                progress.mark_skipped(
                    completed=phase_offsets["predict"] + time_index + 1,
                    total=progress_total,
                    detail=f"{now.strftime('%Y-%m-%d')} predict exists",
                )
                continue
            predictor = Predictor(
                config,
                now,
                ensemble,
                progress=progress,
                time_index=time_index,
                time_total=len(time_points),
                phase_offset=phase_offsets["predict"],
                progress_total=progress_total,
            )
            dataloader = FinancialDataloader(config, now, X, y)
            predictor.predict(dataloader)
            del predictor
            empty_device_cache(args.device)

    if active_phases:
        progress.mark_done()

    if args.econ:
        from run.econ import EconEvaluator
        econ = EconEvaluator(config)

    if args.group_ablation:
        from run.econ import GroupAblationEvaluator
        group_ablation = GroupAblationEvaluator(config)
