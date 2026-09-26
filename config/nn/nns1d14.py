from config.base import Config, IOConfig
from config.seeded_static import expand_widths, pair_hidden_sizes
from utils.logger import Logger


class NNS1D14Config(Config):
    def __init__(self, args):
        super().__init__(args)
        self.model.name = 'nns1d14'
        self.model.type = 'NN'
        self.model.hidden_sizes = expand_widths(seed=1, depth=14)
        self.io = IOConfig(model_name=self.model.name, asset_prefix=args.asset_prefix)
        self.logger = Logger(user_level='info', log_dir=self.io.log_dir)
        self.logger.set_logger()
