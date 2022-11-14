import torch.nn as nn
import torch.nn.functional as F

from due.layers import spectral_norm_fc


class FCResNet(nn.Module):
    def __init__(
        self,
        preprocess,
        features,
        depth,
        spectral_normalization,
        coeff=0.95,
        n_power_iterations=1,
        dropout_rate=0.01,
        num_outputs=None,
        activation="relu",
    ):
        super().__init__()
        """
        ResFNN architecture

        Introduced in SNGP: https://arxiv.org/abs/2006.10108
        """
        self.preprocess = preprocess
        self.first =  nn.Linear(features, features)
        self.residuals = nn.ModuleList(
            [nn.Linear(features, features) for i in range(depth)]
        )
        self.dropout = nn.Dropout(dropout_rate)

        if spectral_normalization:
            self.first = spectral_norm_fc(
                self.first, coeff=coeff, n_power_iterations=n_power_iterations
            )

            for i in range(len(self.residuals)):
                self.residuals[i] = spectral_norm_fc(
                    self.residuals[i],
                    coeff=coeff,
                    n_power_iterations=n_power_iterations,
                )

        self.num_outputs = num_outputs
        if num_outputs is not None:
            self.last = nn.Linear(features, num_outputs)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "elu":
            self.activation = F.elu
        else:
            raise ValueError("That acivation is unknown")

    def apply(self, fn):
        for name, module in self.named_children():
            if name == "preprocess":
                continue
            module.apply(fn)
        return super().apply(fn)

    def _apply(self, fn):
        for name, module in self.named_children():
            if name == "preprocess":
                continue
            module._apply(fn)
        return self

    def _recurse_update_dict(self, submodule, destination, info):
        return 

    def _recurse_update_dict(submodule, destination, info):
        return


    def forward(self, x):
        x = self.preprocess(x)
        x = self.first(x.cuda())

        for residual in self.residuals:
            x = x + self.dropout(self.activation(residual(x)))

        if self.num_outputs is not None:
            x = self.last(x)

        return x
