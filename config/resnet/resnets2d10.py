from config.base import Config, IOConfig
from config.seeded_static import expand_widths, pair_hidden_sizes
from utils.logger import Logger


class ResNetS2D10Config(Config):
    def __init__(self, args):
        super().__init__(args)
        self.model.name = 'resnets2d10'
        self.model.type = 'ResNet'
        self.model.hidden_sizes = expand_widths(seed=2, depth=10)
        self.io = IOConfig(model_name=self.model.name, asset_prefix=args.asset_prefix)
        self.logger = Logger(user_level='info', log_dir=self.io.log_dir)
        self.logger.set_logger()
