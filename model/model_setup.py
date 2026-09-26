import torch
from model.nn import NN
from model.nnp import NNp
from model.resnet import ResNet
from model.resnetp import ResNetp

class ModelSetup:
    def __init__(self, config):
        self.config = config
        self._create_model()

    def _input_features(self):
        return self.config.data.number_of_chars + self.config.data.number_of_macro

    def _resolved_hidden_sizes(self):
        hidden_sizes = self.config.model.hidden_sizes

        if self.config.model.type not in {"ResNet+", "NN+"}:
            return hidden_sizes

        expected_input = self._input_features()
        if not hidden_sizes:
            return hidden_sizes

        first_input = hidden_sizes[0][0]
        if first_input == expected_input:
            return hidden_sizes

        resolved_hidden_sizes = [
            [expected_input if in_sz == first_input else in_sz, out_sz]
            for in_sz, out_sz in hidden_sizes
        ]
        self.config.logger.warning(
            f"Adjusted model hidden_sizes input dimension from {first_input} to {expected_input} for {self.config.model.name}"
        )
        return resolved_hidden_sizes

    def _create_model(self):
        input_features = self._input_features()
        hidden_sizes = self._resolved_hidden_sizes()

        if self.config.model.type == 'NN':
            self.config.logger.debug(f"Assigned NN model with {input_features} input features and {self.config.model.hidden_sizes} hidden sizes to device {self.config.training.device}")
            self.model = NN(
                in_features=input_features,
                hidden_sizes=self.config.model.hidden_sizes,
            )
        elif self.config.model.type == 'ResNet':
            self.config.logger.debug(f"Assigned ResNet model with {input_features} input features and {self.config.model.hidden_sizes} hidden sizes to device {self.config.training.device}")
            self.model = ResNet(in_features=input_features,
                                hidden_sizes=self.config.model.hidden_sizes)
        elif self.config.model.type == 'ResNet+':
            self.config.logger.debug(f"Assigned ResNet+ model with {input_features} input features and {hidden_sizes} hidden sizes to device {self.config.training.device}")
            self.model = ResNetp(
                                hidden_sizes=hidden_sizes,
                                dropout=self.config.model.dropout,
                                negative_slope=self.config.model.negative_slope)
        elif self.config.model.type == 'NN+':
            self.config.logger.debug(f"Assigned NN+ model with {input_features} input features and {hidden_sizes} hidden sizes to device {self.config.training.device}")
            self.model = NNp(
                                hidden_sizes=hidden_sizes,
                                dropout=self.config.model.dropout,
                                negative_slope=self.config.model.negative_slope)
        else:
            raise ValueError(f"Model name {self.config.model.name} not supported")

        self.model = self.model.to(self.config.training.device)
        self.config.logger.debug(self.model)
        self.config.logger.debug(f"Total number of parameters: {sum(p.numel() for p in self.model.parameters())}")
