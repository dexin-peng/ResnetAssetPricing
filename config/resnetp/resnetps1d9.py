from config.base import Config, IOConfig
from config.seeded_static import expand_pair_hidden_sizes
from utils.logger import Logger


class ResNetpS1D9Config(Config):
    def __init__(self, args):
        super().__init__(args)
        self.model.name = 'resnetps1d9'
        self.model.type = 'ResNet+'
        self.model.hidden_sizes = expand_pair_hidden_sizes(seed=1, depth=9)
        self.model.dropout = 0.1
        self.model.negative_slope = 0.01
        self.io = IOConfig(model_name=self.model.name, asset_prefix=args.asset_prefix)
        self.logger = Logger(user_level='info', log_dir=self.io.log_dir)
        self.logger.set_logger()
